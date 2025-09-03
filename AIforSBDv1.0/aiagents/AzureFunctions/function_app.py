import azure.functions as func
import logging
import json
import os
import tempfile
import shutil
import subprocess
import re
from typing import Dict, Any
import git
from datetime import datetime
import uuid
import hcl2  # For parsing HCL without terraform binary

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="github-fetcher", methods=["POST"])
async def github_fetcher(req: func.HttpRequest) -> func.HttpResponse:
    """Fetch GitHub repository and analyze Terraform files"""
    try:
        req_body = req.get_json()
        github_url = req_body.get('github_url')
        
        if not github_url:
            return func.HttpResponse(
                json.dumps({"error": "github_url is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="github_")
        
        try:
            # Clone repository
            repo_name = github_url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join(temp_dir, repo_name)
            
            logging.info(f"Cloning repository: {github_url}")
            git.Repo.clone_from(github_url, repo_path, depth=1)
            
            # Find Terraform files
            terraform_files = []
            tf_directories = {}
            tf_file_contents = {}  # Store file contents
            
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if file.endswith('.tf') or file.endswith('.tfvars'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, repo_path)
                        terraform_files.append(rel_path)
                        
                        # Read and store file content
                        with open(full_path, 'r', encoding='utf-8') as f:
                            tf_file_contents[rel_path] = f.read()
                        
                        # Track directories with .tf files
                        if file.endswith('.tf'):
                            dir_path = os.path.dirname(rel_path)
                            tf_directories[dir_path] = tf_directories.get(dir_path, 0) + 1
            
            # Store repo info in blob storage
            storage_connection = os.environ.get("AzureWebJobsStorage")
            container_name = "terraform-repos"
            repo_id = str(uuid.uuid4())
            
            # Upload terraform files as JSON to blob storage
            from azure.storage.blob import BlobServiceClient
            blob_service = BlobServiceClient.from_connection_string(storage_connection)
            
            # Ensure container exists
            container_client = blob_service.get_container_client(container_name)
            if not container_client.exists():
                container_client.create_container()
            
            # Store terraform files content
            repo_data = {
                "github_url": github_url,
                "repo_name": repo_name,
                "terraform_files": terraform_files,
                "tf_file_contents": tf_file_contents,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Upload as JSON
            blob_client = blob_service.get_blob_client(container=container_name, blob=f"{repo_id}.json")
            blob_client.upload_blob(json.dumps(repo_data), overwrite=True)
            
            result = {
                "status": "success",
                "repo_id": repo_id,
                "terraform_files": terraform_files,
                "terraform_directories": [
                    {"path": path, "tf_count": count} 
                    for path, count in tf_directories.items()
                ],
                "total_files": len(terraform_files)
            }
            
            return func.HttpResponse(
                json.dumps(result),
                mimetype="application/json"
            )
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logging.error(f"Error in github_fetcher: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "status": "failed"}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="terraform-analyze", methods=["POST"])
async def terraform_analyze(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        repo_id = req_body.get('repo_id')
        target_directory = req_body.get('target_directory', '')
        tfvars_content = req_body.get('tfvars_content', '')
        variable_overrides = req_body.get('variable_overrides', {})

        if not repo_id:
            return func.HttpResponse(
                json.dumps({"error": "repo_id is required"}),
                status_code=400,
                mimetype="application/json"
            )

        # Download repo data from blob storage
        storage_connection = os.environ.get("AzureWebJobsStorage")
        from azure.storage.blob import BlobServiceClient

        blob_service = BlobServiceClient.from_connection_string(storage_connection)
        blob_client = blob_service.get_blob_client(
            container="terraform-repos", 
            blob=f"{repo_id}.json"
        )

        # Download and parse repo data
        repo_data = json.loads(blob_client.download_blob().readall())
        tf_file_contents = repo_data['tf_file_contents']

        # Log available files
        logging.info(f"Available TF files: {list(tf_file_contents.keys())}")
        logging.info(f"Target directory: '{target_directory}'")

        # Filter files for target directory
        if target_directory:
            filtered_contents = {
                path: content 
                for path, content in tf_file_contents.items() 
                if path.startswith(target_directory)
            }
        else:
            filtered_contents = tf_file_contents

        logging.info(f"Filtered TF files: {list(filtered_contents.keys())}")

        # Parse Terraform files
        parsed_configs = parse_terraform_files(filtered_contents, tfvars_content, variable_overrides)
        
        # Generate plan-like structure
        terraform_json = generate_plan_structure(parsed_configs)
        
        # Create summary
        resources = terraform_json.get('planned_values', {}).get('root_module', {}).get('resources', [])
        
        summary = {
            "total_resources": len(resources),
            "resource_types": {}
        }
        
        for resource in resources:
            res_type = resource.get('type', 'unknown')
            summary['resource_types'][res_type] = summary['resource_types'].get(res_type, 0) + 1
        
        # Upload JSON to blob storage
        json_blob_name = f"plans/{repo_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        json_blob_client = blob_service.get_blob_client(
            container="terraform-plans",
            blob=json_blob_name
        )
        
        # Ensure container exists
        plans_container = blob_service.get_container_client("terraform-plans")
        if not plans_container.exists():
            plans_container.create_container()
        
        json_blob_client.upload_blob(
            json.dumps(terraform_json),
            overwrite=True
        )
        
        # Generate SAS URL
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        from datetime import timedelta
        
        sas_token = generate_blob_sas(
            account_name=blob_service.account_name,
            container_name="terraform-plans",
            blob_name=json_blob_name,
            account_key=blob_service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        
        json_url = f"{json_blob_client.url}?{sas_token}"
        
        result = {
            "status": "success",
            "terraform_json_url": json_url,
            "terraform_json": terraform_json,
            "summary": summary,
            "message": "Terraform configuration analyzed successfully",
            "analysis_type": "static"  # Indicate this is static analysis
        }
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in terraform_analyze: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "status": "failed"}),
            status_code=500,
            mimetype="application/json"
        )

def parse_terraform_files(tf_files: Dict[str, str], tfvars_content: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Terraform files and extract configuration (v2-normalize)"""
    logging.info("Parser version: v2-normalize")

    resources = []
    variables = {}
    providers = {}
    data_sources = []

    # Parse tfvars if provided
    tfvars = {}
    if tfvars_content:
        try:
            tfvars = hcl2.loads(tfvars_content)
        except Exception as e:
            logging.warning(f"Failed to parse tfvars content: {e}")

    def normalize_block(block):
        """
        Convert python-hcl2 list-of-dicts form into a single dict:
        [ { type1 = {...} }, { type2 = {...} } ]  -> { type1: {...}, type2: {...} }
        """
        if isinstance(block, dict):
            return block
        if isinstance(block, list):
            merged = {}
            for item in block:
                if isinstance(item, dict):
                    for k, v in item.items():
                        # If duplicate keys appear, later wins (simplest behavior)
                        merged[k] = v
            return merged
        return {}

    for filename, content in tf_files.items():
        logging.info(f"[parse] File: {filename}")
        if not filename.endswith(".tf"):
            continue
        try:
            raw = hcl2.loads(content)
            logging.info(f"[parse] Raw type: {type(raw)}")

            # raw can be dict OR list of top-level blocks
            top = {}
            if isinstance(raw, dict):
                top = raw
            elif isinstance(raw, list):
                # Flatten list of top-level dicts
                for entry in raw:
                    if isinstance(entry, dict):
                        for k, v in entry.items():
                            # If key repeats, append/merge by list
                            if k not in top:
                                top[k] = v
                            else:
                                # Convert existing + new to list so none is lost
                                existing = top[k]
                                if not isinstance(existing, list):
                                    existing = [existing]
                                if isinstance(v, list):
                                    existing.extend(v)
                                else:
                                    existing.append(v)
                                top[k] = existing
            else:
                logging.warning(f"[parse] Unexpected top-level type in {filename}: {type(raw)}")
                continue

            # Normalize each known block type
            resource_block = top.get("resource")
            variable_block = top.get("variable")
            provider_block = top.get("provider")
            data_block     = top.get("data")

            resource_block = resource_block if resource_block is not None else {}
            variable_block = variable_block if variable_block is not None else {}
            provider_block = provider_block if provider_block is not None else {}
            data_block     = data_block if data_block is not None else {}

            # Resources: resource_block may be list -> list of dicts -> each dict { type: { name: config } }
            if isinstance(resource_block, list):
                for rdict in resource_block:
                    if isinstance(rdict, dict):
                        for rtype, name_map in rdict.items():
                            nm_norm = normalize_block(name_map)
                            for rname, rconfig in nm_norm.items():
                                cfg_vars = substitute_variables(rconfig, {**variables, **tfvars, **overrides})
                                resources.append({
                                    "address": f"{rtype}.{rname}",
                                    "mode": "managed",
                                    "type": rtype,
                                    "name": rname,
                                    "provider_name": get_provider_from_type(rtype),
                                    "values": cfg_vars
                                })
            elif isinstance(resource_block, dict):
                # Might already be { type: { name: config } }
                for rtype, name_map in resource_block.items():
                    nm_norm = normalize_block(name_map)
                    for rname, rconfig in nm_norm.items():
                        cfg_vars = substitute_variables(rconfig, {**variables, **tfvars, **overrides})
                        resources.append({
                            "address": f"{rtype}.{rname}",
                            "mode": "managed",
                            "type": rtype,
                            "name": rname,
                            "provider_name": get_provider_from_type(rtype),
                            "values": cfg_vars
                        })

            # Variables
            if isinstance(variable_block, list):
                for vdict in variable_block:
                    if isinstance(vdict, dict):
                        for vname, vconfig in vdict.items():
                            if isinstance(vconfig, dict):
                                default = vconfig.get("default")
                                variables[vname] = overrides.get(vname, default if default is not None else f"${{{vname}}}")
            elif isinstance(variable_block, dict):
                for vname, vconfig in variable_block.items():
                    if isinstance(vconfig, dict):
                        default = vconfig.get("default")
                        variables[vname] = overrides.get(vname, default if default is not None else f"${{{vname}}}")

            # Providers
            if isinstance(provider_block, list):
                for pdict in provider_block:
                    if isinstance(pdict, dict):
                        for pname, pconfig in pdict.items():
                            providers[pname] = pconfig
            elif isinstance(provider_block, dict):
                for pname, pconfig in provider_block.items():
                    providers[pname] = pconfig

            # Data sources
            if isinstance(data_block, list):
                for ddict in data_block:
                    if isinstance(ddict, dict):
                        for dtype, name_map in ddict.items():
                            nm_norm = normalize_block(name_map)
                            for dname, dconfig in nm_norm.items():
                                data_sources.append({
                                    "address": f"data.{dtype}.{dname}",
                                    "mode": "data",
                                    "type": dtype,
                                    "name": dname,
                                    "provider_name": get_provider_from_type(dtype),
                                    "values": dconfig
                                })
            elif isinstance(data_block, dict):
                for dtype, name_map in data_block.items():
                    nm_norm = normalize_block(name_map)
                    for dname, dconfig in nm_norm.items():
                        data_sources.append({
                            "address": f"data.{dtype}.{dname}",
                            "mode": "data",
                            "type": dtype,
                            "name": dname,
                            "provider_name": get_provider_from_type(dtype),
                            "values": dconfig
                        })

        except Exception as e:
            logging.error(f"[parse] Exception in {filename}: {e}")
            import traceback
            logging.error(traceback.format_exc())

    logging.info(f"[parse] Summary -> resources:{len(resources)} vars:{len(variables)} providers:{len(providers)} data:{len(data_sources)}")
    return {
        "resources": resources,
        "data_sources": data_sources,
        "variables": variables,
        "providers": providers
    }

def generate_plan_structure(parsed_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a Terraform plan-like structure from parsed configurations"""
    resources = parsed_configs.get('resources', [])
    data_sources = parsed_configs.get('data_sources', [])
    
    # Create plan structure similar to terraform show -json
    plan_structure = {
        "format_version": "1.2",
        "terraform_version": "1.5.0",  # Simulated version
        "planned_values": {
            "root_module": {
                "resources": resources,
                "data_sources": data_sources
            }
        },
        "resource_changes": [
            {
                **resource,
                "change": {
                    "actions": ["create"],
                    "before": None,
                    "after": resource.get("values", {}),
                    "after_unknown": {}
                }
            }
            for resource in resources
        ],
        "configuration": {
            "provider_config": parsed_configs.get('providers', {}),
            "root_module": {
                "resources": group_resources_by_type(resources),
                "data_sources": group_resources_by_type(data_sources),
                "variables": parsed_configs.get('variables', {})
            }
        }
    }
    
    return plan_structure

def substitute_variables(config: Any, variables: Dict[str, Any]) -> Any:
    """Recursively substitute variable references in configuration"""
    if isinstance(config, dict):
        return {k: substitute_variables(v, variables) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_variables(item, variables) for item in config]
    elif isinstance(config, str):
        # Simple variable substitution (not full HCL interpolation)
        for var_name, var_value in variables.items():
            config = config.replace(f"${{var.{var_name}}}", str(var_value))
        return config
    else:
        return config

def get_provider_from_type(resource_type: str) -> str:
    """Determine provider from resource type"""
    if resource_type.startswith("azurerm_"):
        return "azurerm"
    elif resource_type.startswith("aws_"):
        return "aws"
    elif resource_type.startswith("google_"):
        return "google"
    else:
        return resource_type.split("_")[0]

def group_resources_by_type(resources: list) -> Dict[str, Dict[str, Any]]:
    """Group resources by type for configuration section"""
    grouped = {}
    for resource in resources:
        res_type = resource['type']
        res_name = resource['name']
        if res_type not in grouped:
            grouped[res_type] = {}
        grouped[res_type][res_name] = resource.get('values', {})
    return grouped