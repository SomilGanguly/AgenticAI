from typing import Optional, List, Dict, Any, Set

import argparse
import json
import logging
import os
import sys
import aiohttp
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
try:
    from azure.keyvault.secrets import SecretClient
except Exception:
    SecretClient = None
import asyncio
import re
import hashlib

from dotenv import load_dotenv

from logging_config import configure_logging
from intake_agent import ensure_agent
from asr_agent_latest import run_asr_agent


load_dotenv()
logger = configure_logging(os.getenv("APP_LOG_FILE", ""))
if os.getenv("APP_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on", "debug"}:
    logger.debug("Verbose logging enabled (APP_VERBOSE=true)")


def check_index_exists(index_name: str) -> Optional[bool]:
    """Return True if index exists, False if not, None if cannot verify.

    Uses AZURE_SEARCH_SERVICE_ENDPOINT and prefers AZURE_SEARCH_API_KEY; else AAD.
    """
    try:
        svc_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        if not svc_endpoint:
            logger.warning("AZURE_SEARCH_SERVICE_ENDPOINT is not set; cannot verify index existence.")
            return None

        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        from azure.search.documents.indexes import SearchIndexClient
        if api_key:
            from azure.core.credentials import AzureKeyCredential
            credential = AzureKeyCredential(api_key)
        else:
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            credential = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)

        sic = SearchIndexClient(endpoint=svc_endpoint, credential=credential)
        sic.get_index(index_name)
        return True
    except Exception as ex:
        if ex.__class__.__name__ == "ResourceNotFoundError":
            return False
        logger.warning("Index existence check failed: %s", ex)
        return None


def _sanitize_index_name(app_id: str) -> str:
    """Convert arbitrary appId to a valid Azure AI Search index name.
    
    Rules: lowercase, alphanumerics or dashes; must start/end with alphanumeric; length 2-128.
    Matches the same logic used in the indexer function.
    """
    import re
    import hashlib
    
    base = app_id.lower().strip()
    base = re.sub(r"[^a-z0-9-]", "-", base)            # invalid chars -> dash
    base = re.sub(r"-+", "-", base)                     # collapse dashes
    base = base.strip("-")                              # trim
    if not base or not base[0].isalnum():
        base = f"app-{hashlib.sha1(app_id.encode()).hexdigest()[:8]}"
    if len(base) < 2:
        base = (base + "ix")[:2]
    if len(base) > 128:
        base = base[:128]
    return base


def delete_search_index(app_id: str) -> bool:
    """Delete the search index for the given application ID.
    
    Returns True if successfully deleted, False otherwise.
    Uses the same naming convention as the indexer function.
    """
    try:
        svc_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
        if not svc_endpoint:
            logger.warning("AZURE_SEARCH_SERVICE_ENDPOINT (or AZURE_SEARCH_ENDPOINT) is not set; cannot delete index.")
            return False

        api_key = os.getenv("AZURE_SEARCH_API_KEY") or os.getenv("AZURE_SEARCH_ADMIN_KEY")
        from azure.search.documents.indexes import SearchIndexClient
        
        if api_key:
            from azure.core.credentials import AzureKeyCredential
            credential = AzureKeyCredential(api_key)
        else:
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            credential = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)

        # Use the same naming logic as the indexer
        index_name = _sanitize_index_name(app_id)
        
        sic = SearchIndexClient(endpoint=svc_endpoint, credential=credential)
        
        # Check if index exists before attempting to delete
        try:
            sic.get_index(index_name)
        except Exception as ex:
            if ex.__class__.__name__ == "ResourceNotFoundError":
                logger.debug(f"Index '{index_name}' does not exist, nothing to delete")
                return True  # Consider this a success since the goal is achieved
            else:
                logger.error(f"Error checking index existence: {ex}")
                return False
        
        # Delete the index
        sic.delete_index(index_name)
        logger.debug(f"Successfully deleted search index: {index_name}")
        return True
        
    except Exception as ex:
        logger.error(f"Failed to delete search index for app_id '{app_id}': {ex}")
        return False


def _get_kv_secret(secret_name: str) -> Optional[str]:
    """Retrieve a single secret value from Key Vault using DefaultAzureCredential.

    Expects KEYVAULT_URL in environment. Returns None if not available or on failure.
    """
    try:
        if not SecretClient:
            logger.debug("SecretClient unavailable (azure-keyvault-secrets not installed)")
            return None
        kv_url = os.getenv("KEYVAULT_URL") or os.getenv("KEY_VAULT_URL")
        if not kv_url:
            logger.debug("KEYVAULT_URL not set; skipping Key Vault secret retrieval for %s", secret_name)
            return None
        cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
        sc = SecretClient(vault_url=kv_url, credential=cred)
        return sc.get_secret(secret_name).value
    except Exception as ex:
        logger.warning(f"Key Vault secret fetch failed for {secret_name}: {ex}")
        return None

async def trigger_indexing_function(app_id: str, container_name: str) -> Dict[str, Any]:
    """Trigger the indexing function app and return the result.

    Resolution order for function key:
      1. AZURE_INDEXING_FUNCTION_KEY env var
      2. Secret in Key Vault named INDEXING-FUNCTION-KEY (configurable via AZURE_INDEXING_FUNCTION_KEY_SECRET_NAME)
      3. No key (public function) if neither available.
    """
    function_url = os.getenv("AZURE_INDEXING_FUNCTION_URL")
    if not function_url:
        raise ValueError("AZURE_INDEXING_FUNCTION_URL not set")

    # Get key from env or Key Vault
    function_key = os.getenv("AZURE_INDEXING_FUNCTION_KEY")
    if not function_key:
        secret_name = os.getenv("AZURE_INDEXING_FUNCTION_KEY_SECRET_NAME", "INDEXING-FUNCTION-KEY")
        function_key = _get_kv_secret(secret_name)
        if function_key:
            logger.debug(f"Retrieved function key from Key Vault secret '{secret_name}'")
        else:
            logger.debug("No function key found in env or Key Vault; invoking without key header")

    headers = {"Content-Type": "application/json"}
    if function_key:
        headers["x-functions-key"] = function_key

    payload = {"appId": app_id, "container": container_name}

    logger.debug(f"Triggering indexing function for appId={app_id}, container={container_name}")
    logger.debug(f"Function URL: {function_url}")

    async with aiohttp.ClientSession() as session:
        async with session.post(function_url, json=payload, headers=headers) as response:
            try:
                result = await response.json()
            except Exception:
                text = await response.text()
                result = {"status": "error", "raw": text, "http_status": response.status}
            logger.debug(f"Indexing function response ({response.status}): {result}")
            return result


async def chat_loop(orchestrator_name: str, application_id: Optional[str] = None, cleanup: bool = True) -> None:
    """Interactive chat using Semantic Kernel AzureAIAgent, with /create fallback."""
    logger.debug("Initializing SK AzureAIAgent and credentials")
    try:
        from azure.identity.aio import DefaultAzureCredential
        from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
    except Exception:
        print("semantic-kernel is not installed. Install with: pip install semantic-kernel", flush=True)
        sys.exit(1)

    # Resolve endpoint and deployment from SK settings or env fallbacks
    settings = None
    logger.debug("Resolving AzureAIAgentSettings")
    try:
        settings = AzureAIAgentSettings()
    except Exception:
        settings = None

    endpoint = None
    deployment = None
    if settings:
        endpoint = getattr(settings, "endpoint", None) or os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
        deployment = (
            getattr(settings, "model_deployment_name", None)
            or getattr(settings, "deployment_name", None)
            or os.getenv("AZURE_AI_AGENT_DEPLOYMENT_NAME")
        )
    else:
        endpoint = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
        deployment = os.getenv("AZURE_AI_AGENT_DEPLOYMENT_NAME")

    logger.debug(f"Resolved endpoint={endpoint}, deployment={deployment}")
    if not endpoint or not deployment:
        print("Missing project endpoint or model deployment. Set AZURE_EXISTING_AIPROJECT_ENDPOINT and AZURE_AI_AGENT_DEPLOYMENT_NAME.", flush=True)
        sys.exit(1)

    # Prompt for application id if not provided
    if not application_id:
        try:
            application_id = input("Enter Application ID: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            application_id = None
        if not application_id:
            print("Application ID is required.", flush=True)
            sys.exit(1)
    logger.debug(f"Using Application ID: {application_id}")

    from semantic_kernel.functions import kernel_function

    # Track threads more efficiently - one per agent
    agent_threads: Dict[str, str] = {}  # agent_id -> thread_id mapping
    answer_agent_thread_id: Optional[str] = None  # Single persistent thread for answer agent
    
    # Track the most recent answer-agent thread id for diagnostics
    last_answer_thread_id: Optional[str] = None
    # Track low confidence questions for interactive resolution
    low_confidence_questions: List[Dict[str, Any]] = []
    # Track ephemeral threads created via direct AIProjectClient usage
    ephemeral_thread_ids: Set[str] = set()
    # Track any answer (intake) agent IDs created/retrieved so we can optionally clean them up
    answer_agent_ids: Set[str] = set()

    class OrchestratorPlugin:
        @kernel_function(description="Check if an Azure AI Search index exists for this application.")
        def check_index(self, agent_name: Optional[str] = None, index_name: Optional[str] = None) -> str:
            name = agent_name or application_id
            idx = index_name or os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", name)
            logger.debug(f"Checking index existence for index='{idx}'")
            st = check_index_exists(idx)
            return json.dumps({"index": idx, "exists": st is True, "verifiable": st is not None})

        @kernel_function(description="Ensure an Azure Blob Storage container exists (create if missing). Uses AZURE_STORAGE_CONNECTION_STRING or DefaultAzureCredential with AZURE_STORAGE_ACCOUNT_URL.")
        def check_container_exists(self, container_name: Optional[str] = None, account_url: Optional[str] = None) -> str:
            try:
                container = container_name or application_id
                from azure.storage.blob import BlobServiceClient
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    bsc = BlobServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    acct_url = account_url or os.getenv("AZURE_STORAGE_ACCOUNT_URL")
                    if not acct_url:
                        return json.dumps({"result": "unverified", "reason": "Missing AZURE_STORAGE_ACCOUNT_URL"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    bsc = BlobServiceClient(account_url=acct_url, credential=cred)
                cc = bsc.get_container_client(container)
                try:
                    cc.get_container_properties()
                    return json.dumps({"container": container, "exists": True, "created": False, "url": getattr(cc, "url", None)})
                except Exception as ex:
                    if ex.__class__.__name__ in {"ResourceNotFoundError", "ResourceNotFound"}:
                        try:
                            cc.create_container()
                            return json.dumps({"container": container, "exists": True, "created": True, "url": getattr(cc, "url", None), "message": f"Container '{container}' created. Please upload your files."})
                        except Exception as cex:
                            if "ContainerAlreadyExists" in str(cex):
                                return json.dumps({"container": container, "exists": True, "created": False, "url": getattr(cc, "url", None)})
                            return json.dumps({"result": "error", "message": f"Create container failed: {cex}"})
                    return json.dumps({"result": "unverified", "message": str(ex)})
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        @kernel_function(description="Clone template table to create app-specific QA table. Copies all questions from template.")
        def clone_template_table(self, template_table: Optional[str] = None, target_table: Optional[str] = None) -> str:
            try:
                template = template_table or os.getenv("AZURE_QA_TEMPLATE_TABLE", "AppDetailsTemplate")
                target = target_table or f"AppDetails{application_id}"
                from azure.data.tables import TableServiceClient
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        return json.dumps({"result": "error", "reason": "Missing AZURE_TABLES_ACCOUNT_URL"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                try:
                    existing = [t.name for t in tsc.list_tables()]
                    if target in existing:
                        tc_target = tsc.get_table_client(target)
                        try:
                            existing_count = sum(1 for _ in tc_target.query_entities(max_page_size=1000))
                        except Exception:
                            existing_count = 0
                        return json.dumps({"result": "exists", "table": target, "message": f"Table already exists with {existing_count} rows"})
                except Exception:
                    pass
                try:
                    tsc.create_table(target)
                except Exception as e:
                    if "AlreadyExists" not in str(e):
                        return json.dumps({"result": "error", "message": f"Create table failed: {e}"})
                tc_template = tsc.get_table_client(template)
                tc_target = tsc.get_table_client(target)
                copied = 0
                for ent in tc_template.list_entities():
                    new_e = {
                        "PartitionKey": application_id,
                        "RowKey": ent.get("RowKey", f"Q{copied+1:03d}"),
                        "Question": ent.get("Question", ""),
                        "Guidance": ent.get("Guidance", ""),
                        "Response": "",
                        "Confidence": 0.0,
                        "Citation": ""
                    }
                    if new_e["Question"]:
                        tc_target.upsert_entity(new_e)
                        copied += 1
                if copied == 0:
                    return json.dumps({"result": "warning", "template": template, "target": target, "copied": 0, "message": "Template had no questions"})
                return json.dumps({"result": "ok", "template": template, "target": target, "copied": copied})
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        @kernel_function(description="Clone multiple template tables for application setup.")
        def clone_all_templates(self, templates: Optional[List[str]] = None) -> str:
            try:
                default_templates = [
                    "AppDetailsTemplate",
                    "PrivacyAndSecurity",
                    "IntegrationDependencyTemplate",
                    "MsSqlDBTemplate",
                    "OracleDBTemplate",
                    "InfrastructureDetails"
                ]
                templates_to_clone = templates or default_templates
                from azure.data.tables import TableServiceClient
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        return json.dumps({"result": "error", "message": "Missing AZURE_TABLES_ACCOUNT_URL"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                results = {}
                for template in templates_to_clone:
                    if "AppDetails" in template:
                        target = f"AppDetails{application_id}"
                    elif "PrivacyAndSecurity" in template:
                        target = f"PrivacyAndSecurity{application_id}"
                    elif "IntegrationDependency" in template:
                        target = f"IntegrationDependency{application_id}"
                    elif "MSSQLBD" in template or "MSSQLDB" in template:
                        target = f"MSSQLDB{application_id}"
                    elif "OracleDB" in template:
                        target = f"OracleDB{application_id}"
                    elif "InfrastructureDetails" in template:
                        target = f"InfrastructureDetails{application_id}"
                    else:
                        target = f"{template.replace('Template','')}{application_id}"
                    results[template] = self._clone_single_template(tsc, template, target, application_id)
                return json.dumps({"result": "ok", "cloned": results})
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        def _clone_single_template(self, tsc, template: str, target: str, app_id: str) -> dict:
            try:
                try:
                    tc_target = tsc.get_table_client(target)
                    existing_count = sum(1 for _ in tc_target.query_entities(max_page_size=1))
                    if existing_count > 0:
                        return {"status": "exists", "table": target, "rows": existing_count}
                except Exception:
                    pass
                try:
                    tsc.create_table(target)
                except Exception as e:
                    if "AlreadyExists" not in str(e):
                        logger.debug(f"Create table warning: {e}")
                tc_template = tsc.get_table_client(template)
                tc_target = tsc.get_table_client(target)
                copied = 0
                for ent in tc_template.list_entities():
                    ent["PartitionKey"] = app_id
                    if template not in ["IntegrationDependencyTemplate", "InfrastructureDetailsTemplate"]:
                        ent.setdefault("Response", "")
                        ent.setdefault("Confidence", 0.0)
                        ent.setdefault("Citation", "")
                    tc_target.upsert_entity(ent)
                    copied += 1
                return {"status": "created", "table": target, "copied": copied}
            except Exception as ex:
                return {"status": "error", "message": str(ex)}

        @kernel_function(description="Read a field from Azure Table Storage to determine indexing status.")
        def get_indexing_status(self, table_name: Optional[str] = None, partition_key: Optional[str] = None, row_key: Optional[str] = None, field_name: Optional[str] = "IndexingStatus", account_url: Optional[str] = None) -> str:
            try:
                table = table_name or os.getenv("AZURE_INDEX_STATUS_TABLE_NAME") or "indexingStatus"
                pk = partition_key or os.getenv("AZURE_INDEX_STATUS_PARTITION_KEY", "OnPremToAzureMigration")
                rk = row_key or application_id
                logger.debug(f"Indexing status lookup: table={table}, pk={pk}, rk={rk}, field={field_name}")
                
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                try:
                    from azure.data.tables import TableServiceClient
                except Exception as ex:
                    return json.dumps({"result": "error", "message": f"TableServiceClient import failed: {ex}"})

                if conn_str:
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    tables_url = account_url or os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        return json.dumps({"result": "unverified", "reason": "Missing AZURE_TABLES_ACCOUNT_URL"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)

                try:
                    tc = tsc.get_table_client(table_name=table)
                    entity = tc.get_entity(partition_key=pk, row_key=rk)
                    value = entity.get(field_name)
                    return json.dumps({
                        "result": "ok",
                        "table": table,
                        "pk": pk,
                        "rk": rk,
                        "field": field_name,
                        "value": value,
                        "isComplete": bool(value) if isinstance(value, (bool, int, str)) else None,
                    })
                except Exception as ex:
                    if ex.__class__.__name__ in {"ResourceNotFoundError", "ResourceNotFound"}:
                        # Create entity with default values
                        entity_to_create = {
                            "PartitionKey": pk,
                            "RowKey": rk,
                            "IndexingStatus": False,
                            "ContainerCreated": False,
                        }
                        tc = tsc.get_table_client(table_name=table)
                        tc.upsert_entity(entity=entity_to_create)
                        return json.dumps({
                            "result": "created",
                            "table": table,
                            "pk": pk,
                            "rk": rk,
                            "field": field_name,
                            "value": False,
                            "isComplete": False,
                        })
                    return json.dumps({"result": "unverified", "message": str(ex)})
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        @kernel_function(description="Summarize pending Q&A rows for this application.")
        def get_qa_summary(self, table_name: Optional[str] = None, partition_key: Optional[str] = None, account_url: Optional[str] = None) -> str:
            try:
                qa_table = table_name or f"AppDetails{application_id}"
                qa_pk = partition_key or application_id
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                
                try:
                    from azure.data.tables import TableServiceClient
                except Exception as ex:
                    return json.dumps({"result": "error", "message": f"TableServiceClient import failed: {ex}"})
                
                if conn_str:
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    tables_url = account_url or os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        return json.dumps({"result": "unverified", "reason": "Missing AZURE_TABLES_ACCOUNT_URL"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                
                try:
                    tc = tsc.get_table_client(table_name=qa_table)
                    escaped_pk = str(qa_pk).replace("'", "''")
                    server_filter = f"PartitionKey eq '{escaped_pk}'"
                    entities = list(tc.query_entities(query_filter=server_filter))
                    
                    pending = 0
                    low_conf = 0
                    for e in entities:
                        if not e.get("Response") or str(e.get("Response")).strip() in ["", "-"]:
                            pending += 1
                        elif e.get("Confidence", 1.0) < 0.5:
                            low_conf += 1
                    
                    return json.dumps({
                        "result": "ok",
                        "table": qa_table,
                        "pk": qa_pk,
                        "total": len(entities),
                        "pending": pending,
                        "lowConfidence": low_conf
                    })
                except Exception as ex:
                    return json.dumps({"result": "error", "message": str(ex)})
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        @kernel_function(description="Extract and populate integration dependency information.")
        async def populate_dependency_table(self, table_name: Optional[str] = None) -> str:
            """Extract dependency information from search index and populate the table."""
            try:
                from azure.data.tables import TableServiceClient, UpdateMode as TableUpdateMode
                from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                
                dep_table = table_name or f"IntegrationDependency{application_id}"
                agent_id = await ensure_agent(application_id)
                # Get table client
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        return json.dumps({"result": "error", "message": "Missing table storage config"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                
                tc = tsc.get_table_client(table_name=dep_table)
                
                # First, get unique source servers from the search index
                unique_servers = await self._get_unique_servers_from_index(agent_id)
                
                if not unique_servers:
                    return json.dumps({"result": "warning", "message": "No dependency data found in index"})
                
                # Process each unique server
                populated_count = 0
                for server_info in unique_servers:
                    # Generate specialized query for this server's dependencies
                    dep_query = self._generate_dependency_query(server_info)
                    
                    # Get dependency information from the agent
                    dep_info = await self._query_dependencies_for_server(dep_query, server_info, agent_id)
                    
                    if dep_info and isinstance(dep_info, list):
                        for dep in dep_info:
                            # Create entity for table
                            entity = {
                                "PartitionKey": application_id,
                                "RowKey": f"{dep.get('SourceHostname', 'unknown')}_{dep.get('DestinationHostname', 'unknown')}_{populated_count}",
                                "SourceHostname": dep.get("SourceHostname", ""),
                                "SourceIPAddress": dep.get("SourceIPAddress", ""),
                                "DestinationHostname": dep.get("DestinationHostname", ""),
                                "DestinationIPAddress": dep.get("DestinationIPAddress", ""),
                                "InboundOrOutboundProtocol": dep.get("Protocol", ""),
                                "InboundOrOutboundPortNumber": dep.get("Port", ""),
                                "Description": dep.get("Description", ""),
                                "Confidence": dep.get("Confidence", 0.0),
                                "Citation": dep.get("Citation", "")
                            }
                            
                            tc.upsert_entity(entity=entity, mode=TableUpdateMode.REPLACE)
                            populated_count += 1
                
                return json.dumps({
                    "result": "ok",
                    "table": dep_table,
                    "populated": populated_count,
                    "servers_processed": len(unique_servers)
                })
                
            except Exception as ex:
                logger.exception(f"Failed to populate dependency table: {ex}")
                return json.dumps({"result": "error", "message": str(ex)})

        async def _get_unique_servers_from_index(self, agent_id: str) -> List[str]:
            """Query the agent to get unique servers from the dependency data."""
            try:
                if not agent_id:
                    logger.warning("No agent available to query for servers")
                    return []
                
                # Query to extract server names
                server_query = """Analyze all the dependency tables and network connection information in the indexed documents.  
                  
                Use azure AI search tool.    
                Extract and list ALL unique server names (hostnames) that appear in the data.  
                Include servers that appear as either source or destination.  
                  
                Return ONLY a JSON array of unique server names. Do not include IP addresses.       
                  
                Important:  
                - Include ALL unique server names found in the index  
                - Do NOT include IP addresses  
                - Return only the JSON array, no additional text"""
                
                # Query the agent
                from azure.ai.projects.aio import AIProjectClient
                from azure.identity.aio import DefaultAzureCredential
                
                endpoint = os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
                
                async with DefaultAzureCredential(exclude_shared_token_cache_credential=True) as creds:
                    async with AIProjectClient(credential=creds, endpoint=endpoint) as ai_client:
                        # Reuse thread for this agent
                        thread_id = agent_threads.get(agent_id)
                        if not thread_id:
                            thread = await ai_client.agents.threads.create()
                            thread_id = thread.id
                            agent_threads[agent_id] = thread_id
                            logger.debug(f"Created persistent thread {thread_id} for agent {agent_id}")
                        
                        # Send the query using existing thread
                        await ai_client.agents.messages.create(
                            thread_id=thread_id,
                            role="user",
                            content=server_query
                        )
                        
                        # Run and get response
                        run = await ai_client.agents.runs.create(
                            thread_id=thread_id,
                            agent_id=agent_id
                        )
                        
                        # Wait for completion
                        import asyncio
                        max_wait = 30  # Maximum 30 seconds
                        wait_time = 0
                        while run.status in ["queued", "in_progress", "requires_action"] and wait_time < max_wait:
                            await asyncio.sleep(1)
                            wait_time += 1
                            run = await ai_client.agents.runs.get(
                                thread_id=thread_id,
                                run_id=run.id
                            )
                        
                        if run.status == "completed":
                            # Get messages
                            messages = ai_client.agents.messages.list(thread_id=thread_id)
                            
                            # Parse the assistant's response
                            async for message in messages:
                                if message.role == "assistant":
                                    content = message.content[0].text.value if message.content else ""
                                    
                                    # Try to parse as JSON array
                                    try:
                                        import json, re
                                        # Extract JSON array from the response
                                        if "[" in content and "]" in content:
                                            # Find the JSON array part
                                            start_idx = content.find("[")
                                            end_idx = content.rfind("]") + 1
                                            json_part = content[start_idx:end_idx]
                                            servers = json.loads(json_part)
                                            
                                            if isinstance(servers, list):
                                                # Filter out any IP addresses that might have been included
                                                filtered_servers = [                           
                                                    s.strip() for s in servers
                                                    if isinstance(s, str) and not re.match(r'^\d+\.\d+\.\d+\.\d+$', s)
                                                ]

                                                logger.debug(f"Found {len(filtered_servers)} unique servers from agent: {filtered_servers}")
                                                return filtered_servers
                                    except Exception as parse_ex:
                                        logger.error(f"Failed to parse server list from agent response: {parse_ex}")
                                        logger.debug(f"Response content: {content[:500]}")
                        
                        logger.warning("Could not extract server list from agent")
                        return []
                
            except Exception as ex:
                logger.error(f"Failed to get unique servers from agent: {ex}")
                return []
        def _generate_dependency_query(self, server_name: str) -> str:
            """Generate a specialized query for extracting dependency information for a specific server."""
            return f"""Extract ALL network dependency information for server '{server_name}' from the indexed documents.

Analyze the dependency table/matrix and find ALL connections where '{server_name}' appears as either source or destination.

For EACH connection involving '{server_name}', provide:
- Source Server Name (e.g., POS System, API-GW-01, APP-01)
- Source IP Address (e.g., 192.168.50.10)
- Destination Server Name (e.g., API-GW-01, DB-PRI)
- Destination IP Address (e.g., 10.10.2.10)
- Port Number (e.g., 443, 1433, 8080)
- Protocol (e.g., TCP, UDP, HTTP)
- Description/Purpose (what the connection is used for)

Return the results as a JSON array. Each element should have exactly these fields:
{{
    "SourceHostname": "exact source server name",
    "SourceIPAddress": "source IP",
    "DestinationHostname": "exact destination server name",
    "DestinationIPAddress": "destination IP",
    "Protocol": "TCP/UDP/HTTP/HTTPS",
    "Port": "port number only",
    "Description": "purpose of connection",
    "Confidence": 0.0-1.0,
    "Citation": "source document reference"
}}

Example based on the dependency table:
[
    {{
        "SourceHostname": "POS System",
        "SourceIPAddress": "192.168.50.10",
        "DestinationHostname": "API-GW-01",
        "DestinationIPAddress": "10.10.2.10",
        "Protocol": "TCP",
        "Port": "443",
        "Description": "POS to API Gateway - Secure traffic",
        "Confidence": 0.95,
        "Citation": "Dependency Matrix Sheet1"
    }}
]

IMPORTANT:
- Include ALL connections where '{server_name}' is involved
- Parse the dependency table/matrix correctly
- Return empty array [] if no dependencies found for this server
- Extract actual server names, not generic descriptions"""

        async def _query_dependencies_for_server(self, query: str, server_name: str, agent_id: str) -> List[Dict]:
            """Query the agent for dependency information about a specific server."""
            try:
                if not agent_id:
                    return []
                from azure.ai.projects.aio import AIProjectClient
                from azure.identity.aio import DefaultAzureCredential
                endpoint = os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
                
                async with DefaultAzureCredential(exclude_shared_token_cache_credential=True) as creds:
                    async with AIProjectClient(credential=creds, endpoint=endpoint) as ai_client:
                        # Reuse existing thread for this agent if available
                        thread_id = agent_threads.get(agent_id)
                        if not thread_id:
                            # Create thread only if doesn't exist for this agent
                            thread = await ai_client.agents.threads.create()
                            thread_id = thread.id
                            agent_threads[agent_id] = thread_id
                            logger.debug(f"Created persistent thread {thread_id} for agent {agent_id}")
                        else:
                            logger.debug(f"Reusing thread {thread_id} for agent {agent_id}")
                        
                        try:
                            # Skip message clearing to avoid "message not found" errors
                            # The persistent thread will accumulate context which can be beneficial
                            # for related queries to the same agent
                            
                            await ai_client.agents.messages.create(
                                thread_id=thread_id, 
                                role="user", 
                                content=query
                            )
                            run = await ai_client.agents.runs.create(
                                thread_id=thread_id, 
                                agent_id=agent_id
                            )
                            
                            import asyncio
                            while run.status in ["queued", "in_progress", "requires_action"]:
                                await asyncio.sleep(1)
                                run = await ai_client.agents.runs.get(
                                    thread_id=thread_id, 
                                    run_id=run.id
                                )
                            
                            if run.status == "completed":
                                messages = ai_client.agents.messages.list(thread_id=thread_id)
                                async for message in messages:
                                    if message.role == "assistant":
                                        content = message.content[0].text.value if message.content else ""
                                        try:
                                            import json
                                            dependencies = json.loads(content)
                                            if isinstance(dependencies, list):
                                                return dependencies
                                            if isinstance(dependencies, dict) and "Response" in dependencies:
                                                response_text = dependencies.get("Response", "")
                                                try:
                                                    return json.loads(response_text)
                                                except Exception:
                                                    return self._parse_dependency_text(response_text, server_name)
                                        except Exception:
                                            return self._parse_dependency_text(content, server_name)
                        except Exception as ex:
                            logger.error(f"Query failed for {server_name}: {ex}")
                            # Don't delete thread on error - keep for reuse
            except Exception as ex:
                logger.error(f"Failed to query dependencies for {server_name}: {ex}")
            return []

        def _parse_dependency_text(self, text: str, server_name: str) -> List[Dict]:
            """Parse dependency information from text response."""
            dependencies: List[Dict] = []
            import re
            lines = text.split('\n')
            for line in lines:
                if 'Source' in line and 'Destination' in line:
                    continue
                if not line.strip():
                    continue
                pattern = r'([^\d\s][^\t]*?)\s+([\d\.]+)\s+([^\d\s][^\t]*?)\s+([\d\.]+)\s+(\d+)/(TCP|UDP|HTTP|HTTPS)\s*(.*)?'
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    dependencies.append({
                        "SourceHostname": match.group(1).strip(),
                        "SourceIPAddress": match.group(2).strip(),
                        "DestinationHostname": match.group(3).strip(),
                        "DestinationIPAddress": match.group(4).strip(),
                        "Protocol": match.group(6).upper(),
                        "Port": match.group(5),
                        "Description": match.group(7).strip() if match.group(7) else f"Connection from {match.group(1)} to {match.group(3)}",
                        "Confidence": 0.8,
                        "Citation": "Extracted from dependency table"
                    })
            if not dependencies:
                simple_pattern = r"Source[:\s]+([^\s,]+).*?Destination[:\s]+([^\s,]+).*?Port[:\s]+(\d+)"
                matches = re.findall(simple_pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    dependencies.append({
                        "SourceHostname": match[0].strip(),
                        "SourceIPAddress": "",
                        "DestinationHostname": match[1].strip(),
                        "DestinationIPAddress": "",
                        "Protocol": "TCP",
                        "Port": match[2],
                        "Description": f"Connection from {match[0]} to {match[1]}",
                        "Confidence": 0.7,
                        "Citation": "Extracted from search results"
                    })
            return dependencies

        @kernel_function(description="Extract and populate infrastructure details for servers.")
        async def populate_infrastructure_table(self, table_name: Optional[str] = None) -> str:
            """Extract infrastructure information from search index and populate the table."""
            try:
                from azure.data.tables import TableServiceClient, UpdateMode as TableUpdateMode
                from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                
                infra_table = table_name or f"InfrastructureDetails{application_id}"
                agent_id = await ensure_agent(application_id)
                try:
                    answer_agent_ids.add(agent_id)
                except Exception:
                    pass
                
                # Get table client
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        return json.dumps({"result": "error", "message": "Missing table storage config"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                
                tc = tsc.get_table_client(table_name=infra_table)
                
                # Get unique servers from the search index (reuse existing method)
                unique_servers = await self._get_unique_servers_from_index(agent_id)
                
                if not unique_servers:
                    return json.dumps({"result": "warning", "message": "No server data found in index"})
                
                # Process each unique server
                populated_count = 0
                for server_name in unique_servers:
                    # Generate specialized query for this server's infrastructure details
                    infra_query = self._generate_infrastructure_query(server_name)
                    
                    # Get infrastructure information from the agent
                    infra_info = await self._query_infrastructure_for_server(infra_query, server_name, agent_id)
                    
                    if infra_info and isinstance(infra_info, dict):
                        # Create entity for table
                        entity = {
                            "PartitionKey": application_id,
                            "RowKey": f"{server_name}_{populated_count}",
                            "ApplicationName": application_id,
                            "VMHostname": infra_info.get("VMHostname", server_name),
                            "Domain": infra_info.get("Domain", ""),
                            "IPAddress": infra_info.get("IPAddress", ""),
                            "ServerFunction": infra_info.get("ServerFunction", ""),
                            "OnpremSecurityZone": infra_info.get("OnpremSecurityZone", ""),
                            "OperatingSystem": infra_info.get("OperatingSystem", ""),
                            "vCPU": infra_info.get("vCPU", ""),
                            "RAM": infra_info.get("RAM", ""),
                            "DisksAndSize": infra_info.get("DisksAndSize", ""),
                            "LunId": infra_info.get("LunId", ""),
                            "ServerEnvironment": infra_info.get("ServerEnvironment", ""),
                            "GeneralNotes": infra_info.get("GeneralNotes", ""),
                            "Confidence": infra_info.get("Confidence", 0.0),
                            "Citation": infra_info.get("Citation", "")
                        }
                        
                        tc.upsert_entity(entity=entity, mode=TableUpdateMode.REPLACE)
                        populated_count += 1
                
                return json.dumps({
                    "result": "ok",
                    "table": infra_table,
                    "populated": populated_count,
                    "servers_processed": len(unique_servers)
                })
                
            except Exception as ex:
                logger.exception(f"Failed to populate infrastructure table: {ex}")
                return json.dumps({"result": "error", "message": str(ex)})

        def _generate_infrastructure_query(self, server_name: str) -> str:
            """Generate a specialized query for extracting infrastructure information for a specific server."""
            return f"""Extract ALL infrastructure details for server '{server_name}' from the indexed documents.

Find and extract the following information for '{server_name}':
- VM Hostname (FQDN) - Server name given above or Full qualified domain name
- Domain - Domain the server belongs to
- IP Address - IP address(es) assigned
- Server Function - Web, App, DB, or Others
- On-prem Security Zone - TP/TA/TD/E1/E2/E3/Etc
- Operating System - OS type and version
- vCPU - Number of virtual CPUs
- RAM - Memory in GB
- Disks and Size - Storage configuration and sizes in GB
- LUN ID - Storage LUN identifiers
- Server Environment - Dev/Test/SIT/UAT/Non-Prod/Prod
- General Notes - Any additional notes

Return the results as a JSON object with exactly these fields:
{{
    "VMHostname": "server name given above or FQDN",
    "Domain": "domain name",
    "IPAddress": "IP address",
    "ServerFunction": "Web/App/DB/Others",
    "OnpremSecurityZone": "security zone",
    "OperatingSystem": "OS details",
    "vCPU": "number of vCPUs",
    "RAM": "RAM in GB",
    "DisksAndSize": "disk configuration",
    "LunId": "LUN ID",
    "ServerEnvironment": "Dev/Test/SIT/UAT/Non-Prod/Prod",
    "GeneralNotes": "any notes",
    "Confidence": 0.0-1.0,
    "Citation": "source document reference"
}}

IMPORTANT:
- Extract exact values from documents
- If a field is not found, use empty string ""
- Include confidence score based on data completeness
- Return empty object {{}} if no infrastructure data found for this server"""

        async def _query_infrastructure_for_server(self, query: str, server_name: str, agent_id: str) -> Dict:
            """Query the agent for infrastructure information about a specific server."""
            try:
                if not agent_id:
                    return {}
                from azure.ai.projects.aio import AIProjectClient
                from azure.identity.aio import DefaultAzureCredential
                endpoint = os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
                async with DefaultAzureCredential(exclude_shared_token_cache_credential=True) as creds:
                    async with AIProjectClient(credential=creds, endpoint=endpoint) as ai_client:
                        thread_id = agent_threads.get(agent_id)
                        if not thread_id:
                            # Create thread only if doesn't exist for this agent
                            thread = await ai_client.agents.threads.create()
                            thread_id = thread.id
                            agent_threads[agent_id] = thread_id
                            logger.debug(f"Created persistent thread {thread_id} for agent {agent_id}")
                        else:
                            logger.debug(f"Reusing thread {thread_id} for agent {agent_id}")
                        try:
                            await ai_client.agents.messages.create(
                                thread_id=thread_id,
                                role="user",
                                content=query
                            )
                            run = await ai_client.agents.runs.create(
                                thread_id=thread_id,
                                agent_id=agent_id
                            )
                            import asyncio
                            max_wait = 30
                            wait_time = 0
                            while run.status in ["queued", "in_progress", "requires_action"] and wait_time < max_wait:
                                await asyncio.sleep(1)
                                wait_time += 1
                                run = await ai_client.agents.runs.get(
                                    thread_id=thread_id,
                                    run_id=run.id
                                )
                            if run.status == "completed":
                                messages = ai_client.agents.messages.list(thread_id=thread_id)
                                async for message in messages:
                                    if message.role == "assistant":
                                        content = message.content[0].text.value if message.content else ""
                                        try:
                                            import json
                                            infra_data = json.loads(content)
                                            if isinstance(infra_data, dict):
                                                return infra_data
                                        except Exception:
                                            if "{" in content and "}" in content:
                                                try:
                                                    json_part = content[content.find("{"):content.rfind("}")+1]
                                                    return json.loads(json_part)
                                                except Exception:
                                                    pass
                                        return {
                                            "VMHostname": server_name,
                                            "Confidence": 0.3,
                                            "Citation": "Could not parse infrastructure data"
                                        }
                        except Exception as ex:
                            logger.error(f"Query failed for {server_name}: {ex}")
                        # finally:
                        #     try:
                        #         await ai_client.agents.threads.delete(thread_id=thread.id)
                        #     except Exception as _del_ex:
                        #         logger.debug(f"Thread delete failed (infrastructure query {server_name}): {_del_ex}")
            except Exception as ex:
                logger.error(f"Failed to query infrastructure for {server_name}: {ex}")
                return {}

    # Internal helper functions
    async def _update_indexing_status(status: str, container_created: bool = False) -> bool:
        """Update indexing status in the status table."""
        try:
            from azure.data.tables import TableServiceClient
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            
            table = os.getenv("AZURE_INDEX_STATUS_TABLE_NAME") or "indexingStatus"
            pk = os.getenv("AZURE_INDEX_STATUS_PARTITION_KEY", "OnPremToAzureMigration")
            rk = application_id
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            
            if conn_str:
                tsc = TableServiceClient.from_connection_string(conn_str)
            else:
                tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                if not tables_url:
                    return False
                cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                tsc = TableServiceClient(endpoint=tables_url, credential=cred)
            
            tc = tsc.get_table_client(table_name=table)
            
            # Get existing entity or create new
            try:
                entity = tc.get_entity(partition_key=pk, row_key=rk)
            except:
                entity = {
                    "PartitionKey": pk,
                    "RowKey": rk,
                    "IndexingStatus": "pending",
                    "ContainerCreated": False
                }
            
            # Update status
            if status == "true":
                entity["IndexingStatus"] = True
            elif status == "false":
                entity["IndexingStatus"] = False
            else:
                entity["IndexingStatus"] = status  # "pending" or other string
            
            if container_created:
                entity["ContainerCreated"] = True
            
            tc.upsert_entity(entity=entity)
            return True
        except Exception as ex:
            logger.error(f"Failed to update indexing status: {ex}")
            return False

    async def _trigger_and_check_indexing(container_name: str) -> bool:
        """Trigger indexing function and update status based on result."""
        try:
            result = await trigger_indexing_function(application_id, container_name)
            
            if result.get("status") == "ok" and "result" in result:
                uploaded = result["result"].get("uploaded", 0)
                if uploaded >= 1:
                    # Update IndexingStatus to true
                    await _update_indexing_status("true", container_created=True)
                    logger.debug(f"Indexing successful: {uploaded} documents uploaded")
                    return True
                else:
                    # Update IndexingStatus to pending
                    await _update_indexing_status("pending", container_created=True)
                    logger.warning("No documents uploaded during indexing")
                    return False
            else:
                logger.error(f"Indexing failed: {result}")
                return False
        except Exception as ex:
            logger.error(f"Failed to trigger indexing: {ex}")
            return False

                # Add new validation and edit functions
    async def _present_table_results_markdown(table_name: str, entities: list) -> None:
        """Present table results in markdown format for user validation."""
        print(f"\n{'='*80}")
        print(f"## Table: {table_name}")
        print(f"{'='*80}\n")
        
        # Create markdown table
        print("| # | Question | Response | Confidence |")
        print("|---|----------|----------|------------|")
        
        for idx, ent in enumerate(entities, 1):
            question = ent.get("Question", "").replace("|", "\\|")[:80]  # Truncate long questions
            response = ent.get("Response", "").replace("|", "\\|")[:100]  # Truncate long responses
            confidence = ent.get("Confidence", 0.0)
            
            # Add visual indicator for confidence level
            if confidence >= 0.8:
                conf_indicator = f"✓ {confidence:.2f}"
            elif confidence >= 0.5:
                conf_indicator = f"⚠ {confidence:.2f}"
            else:
                conf_indicator = f"✗ {confidence:.2f}"
            
            print(f"| {idx} | {question} | {response} | {conf_indicator} |")
        
        print(f"\n**Total Questions:** {len(entities)}")
        low_conf_count = sum(1 for e in entities if e.get("Confidence", 1.0) < 0.5)
        if low_conf_count > 0:
            print(f"**Low Confidence Answers:** {low_conf_count}")
        print()

    async def _interactive_edit_responses(table_name: str, entities: list, tsc) -> tuple:
        """Allow user to interactively edit responses after validation."""
        tc = tsc.get_table_client(table_name=table_name)
        edited_count = 0
        low_conf_resolved = 0
        
        while True:
            print("\n" + "="*60)
            print("VALIDATION OPTIONS:")
            print("="*60)
            print("1. Approve all responses")
            print("2. Edit specific response(s)")
            print("3. Bulk edit low confidence responses")
            print("4. View full details of a question")
            print("5. Re-display table")
            print("-"*60)
            
            try:
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    print("✓ All responses approved")
                    break
                    
                elif choice == "2":
                    # Edit specific responses
                    edit_nums = input("Enter question numbers to edit (comma-separated, e.g., 1,3,5): ").strip()
                    if not edit_nums:
                        continue
                    
                    try:
                        nums = [int(n.strip()) for n in edit_nums.split(",")]
                        for num in nums:
                            if 1 <= num <= len(entities):
                                ent = entities[num - 1]
                                print(f"\n--- Editing Question #{num} ---")
                                print(f"Question: {ent.get('Question')}")
                                print(f"Current Response: {ent.get('Response', '')}")
                                print(f"Current Confidence: {ent.get('Confidence', 0.0):.2f}")
                                
                                new_response = input("\nEnter new response (or press Enter to keep current): ").strip()
                                if new_response:
                                    # Update in memory
                                    ent["Response"] = new_response
                                    ent["Confidence"] = 0.95  # High confidence for user-edited
                                    ent["Citation"] = "User validated/edited"
                                    
                                    # Update in table
                                    from azure.data.tables import UpdateMode as TableUpdateMode
                                    tc.upsert_entity(entity=ent, mode=TableUpdateMode.MERGE)
                                    edited_count += 1
                                    print(f"✓ Response updated for question #{num}")
                            else:
                                print(f"✗ Invalid question number: {num}")
                    except ValueError:
                        print("✗ Invalid input. Please enter numbers only.")
                        
                elif choice == "3":
                    # Bulk edit low confidence responses
                    low_conf_entities = [(i, e) for i, e in enumerate(entities) 
                                        if e.get("Confidence", 1.0) < 0.5 and e.get("Question")]
                    
                    if not low_conf_entities:
                        print("No low confidence responses to edit.")
                        continue
                    
                    print(f"\n--- Editing {len(low_conf_entities)} Low Confidence Responses ---")
                    for idx, (orig_idx, ent) in enumerate(low_conf_entities, 1):
                        print(f"\nLow Confidence Question {idx}/{len(low_conf_entities)} (#{orig_idx + 1}):")
                        print(f"Q: {ent.get('Question')}")
                        print(f"Current Answer (Confidence: {ent.get('Confidence', 0.0):.2f}): {ent.get('Response', '')}")
                        
                        # Check for conflicts
                        if ent.get("ConflictDetected"):
                            print("⚠ CONFLICT DETECTED! Multiple values found in search index:")
                            conflicting_values = json.loads(ent.get("ConflictingValues", "[]"))
                            for i, value in enumerate(conflicting_values, 1):
                                print(f"  {i}. {value}")
                        
                        new_response = input("\nProvide better answer (or 's' to skip, 'q' to quit editing): ").strip()
                        
                        if new_response.lower() == 'q':
                            break
                        elif new_response.lower() == 's':
                            continue
                        elif new_response:
                            # Update response
                            ent["Response"] = new_response
                            ent["Confidence"] = 0.95
                            ent["Citation"] = "User resolved"
                            
                            # Clear conflict flags if they exist
                            if "ConflictDetected" in ent:
                                del ent["ConflictDetected"]
                            if "ConflictingValues" in ent:
                                del ent["ConflictingValues"]
                            
                            from azure.data.tables import UpdateMode as TableUpdateMode
                            tc.upsert_entity(entity=ent, mode=TableUpdateMode.MERGE)
                            low_conf_resolved += 1
                            print("✓ Response updated")
                    
                elif choice == "4":
                    # View full details
                    try:
                        num = int(input("Enter question number to view details: ").strip())
                        if 1 <= num <= len(entities):
                            ent = entities[num - 1]
                            print(f"\n--- Full Details for Question #{num} ---")
                            print(f"Question: {ent.get('Question')}")
                            print(f"Guidance: {ent.get('Guidance', '')}")
                            print(f"Response: {ent.get('Response', '')}")
                            print(f"Confidence: {ent.get('Confidence', 0.0):.2f}")
                            print(f"Citation: {ent.get('Citation', '')}")
                            if ent.get("ConflictDetected"):
                                print(f"Conflict Detected: Yes")
                                print(f"Conflicting Values: {ent.get('ConflictingValues', '')}")
                        else:
                            print("✗ Invalid question number")
                    except ValueError:
                        print("✗ Please enter a valid number")
                        
                elif choice == "5":
                    # Re-display table
                    await _present_table_results_markdown(table_name, entities)
                    
                else:
                    print("✗ Invalid choice. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting edit mode...")
                break
        
        return edited_count, low_conf_resolved

    async def _process_questions_with_validation(client_obj, agent_id: str, table_name: str, partition_key: str) -> dict:
        """Process questions and allow user validation/editing before finalizing."""
        # First, process all questions (existing logic)
        result = await _process_questions_for_table(client_obj, agent_id, table_name, partition_key)
        
        if result.get("result") != "ok":
            return result
        
        # Now present results for validation
        try:
            from azure.data.tables import TableServiceClient
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if conn_str:
                tsc = TableServiceClient.from_connection_string(conn_str)
            else:
                tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                if not tables_url:
                    return result
                cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                tsc = TableServiceClient(endpoint=tables_url, credential=cred)
            
            tc = tsc.get_table_client(table_name=table_name)
            
            # Get all entities to present
            escaped_pk = str(partition_key).replace("'", "''")
            server_filter = f"PartitionKey eq '{escaped_pk}'"
            entities = list(tc.query_entities(query_filter=server_filter))
            
            # Filter to show only entities with questions
            qa_entities = [e for e in entities if e.get("Question")]
            
            if qa_entities:
                # Present in markdown format
                await _present_table_results_markdown(table_name, qa_entities)
                
                # Ask user if they want to validate/edit
                validate_choice = input("Would you like to validate/edit these responses? (yes/no): ").strip().lower()
                
                if validate_choice in ["yes", "y"]:
                    edited_count, low_conf_resolved = await _interactive_edit_responses(table_name, qa_entities, tsc)
                    
                    # Update result with edit statistics
                    result["edited"] = edited_count
                    result["lowConfidenceResolved"] = low_conf_resolved
                    
                    # Recalculate low confidence count after edits
                    new_low_conf = sum(1 for e in qa_entities if e.get("Confidence", 1.0) < 0.5)
                    result["lowConfidence"] = new_low_conf
                    result["lowConfidenceQuestions"] = [
                        {
                            "Table": table_name,
                            "RowKey": e.get("RowKey"),
                            "Question": e.get("Question"),
                            "Response": e.get("Response", ""),
                            "Confidence": e.get("Confidence", 0.0)
                        }
                        for e in qa_entities if e.get("Confidence", 1.0) < 0.5
                    ]
                    
                    print(f"\n✓ Validation complete for {table_name}")
                    if edited_count > 0:
                        print(f"  - Edited: {edited_count} responses")
                    if low_conf_resolved > 0:
                        print(f"  - Resolved: {low_conf_resolved} low confidence responses")
                else:
                    print(f"✓ Proceeding without validation for {table_name}")
            
        except Exception as ex:
            logger.warning(f"Validation/edit phase failed for {table_name}: {ex}")
            # Continue with original result if validation fails
        
        return result

    async def _present_dependency_results_markdown(table_name: str, entities: list) -> None:
        """Present dependency table results in markdown format."""
        print(f"\n{'='*80}")
        print(f"## Table: {table_name} (Integration Dependencies)")
        print(f"{'='*80}\n")
        
        print("| # | Source | Source IP | Destination | Dest IP | Protocol | Port | Description |")
        print("|---|--------|-----------|-------------|---------|----------|------|-------------|")
        
        for idx, ent in enumerate(entities, 1):
            src_host = (ent.get("SourceHostname", "")[:20] or "-").replace("|", "\\|")
            src_ip = (ent.get("SourceIPAddress", "") or "-").replace("|", "\\|")
            dst_host = (ent.get("DestinationHostname", "")[:20] or "-").replace("|", "\\|")
            dst_ip = (ent.get("DestinationIPAddress", "") or "-").replace("|", "\\|")
            protocol = (ent.get("InboundOrOutboundProtocol", "") or "-").replace("|", "\\|")
            port = (ent.get("InboundOrOutboundPortNumber", "") or "-").replace("|", "\\|")
            desc = (ent.get("Description", "")[:30] or "-").replace("|", "\\|")
            
            print(f"| {idx} | {src_host} | {src_ip} | {dst_host} | {dst_ip} | {protocol} | {port} | {desc} |")
        
        print(f"\n**Total Dependencies:** {len(entities)}")

    async def _interactive_edit_dependencies(table_name: str, entities: list, tsc) -> int:
        """Allow user to edit dependency entries."""
        tc = tsc.get_table_client(table_name=table_name)
        edited_count = 0
        
        while True:
            print("\n" + "="*60)
            print("DEPENDENCY VALIDATION OPTIONS:")
            print("="*60)
            print("1. Approve all entries")
            print("2. Edit specific entry")
            print("3. Add new dependency")
            print("4. Delete entry")
            print("5. Re-display table")
            print("-"*60)
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                print("✓ All dependencies approved")
                break
                
            elif choice == "2":
                try:
                    num = int(input("Enter entry number to edit: ").strip())
                    if 1 <= num <= len(entities):
                        ent = entities[num - 1]
                        print(f"\n--- Editing Entry #{num} ---")
                        print("Leave blank to keep current value")
                        
                        fields = [
                            ("SourceHostname", "Source Hostname"),
                            ("SourceIPAddress", "Source IP"),
                            ("DestinationHostname", "Destination Hostname"),
                            ("DestinationIPAddress", "Destination IP"),
                            ("InboundOrOutboundProtocol", "Protocol"),
                            ("InboundOrOutboundPortNumber", "Port"),
                            ("Description", "Description")
                        ]
                        
                        updated = False
                        for field, label in fields:
                            current = ent.get(field, "")
                            new_val = input(f"{label} [{current}]: ").strip()
                            if new_val:
                                ent[field] = new_val
                                updated = True
                        
                        if updated:
                            from azure.data.tables import UpdateMode as TableUpdateMode
                            tc.upsert_entity(entity=ent, mode=TableUpdateMode.MERGE)
                            edited_count += 1
                            print(f"✓ Entry #{num} updated")
                except ValueError:
                    print("✗ Invalid input")
                    
            elif choice == "3":
                # Add new dependency
                print("\n--- Adding New Dependency ---")
                new_ent = {
                    "PartitionKey": application_id,
                    "RowKey": f"UserAdded_{len(entities)+1}_{edited_count}",
                    "SourceHostname": input("Source Hostname: ").strip(),
                    "SourceIPAddress": input("Source IP: ").strip(),
                    "DestinationHostname": input("Destination Hostname: ").strip(),
                    "DestinationIPAddress": input("Destination IP: ").strip(),
                    "InboundOrOutboundProtocol": input("Protocol (TCP/UDP/HTTP/HTTPS): ").strip(),
                    "InboundOrOutboundPortNumber": input("Port: ").strip(),
                    "Description": input("Description: ").strip(),
                    "Confidence": 1.0,
                    "Citation": "User added"
                }
                tc.upsert_entity(entity=new_ent)
                entities.append(new_ent)
                edited_count += 1
                print("✓ New dependency added")
                
            elif choice == "4":
                try:
                    num = int(input("Enter entry number to delete: ").strip())
                    if 1 <= num <= len(entities):
                        ent = entities[num - 1]
                        confirm = input(f"Delete entry #{num}? (yes/no): ").strip().lower()
                        if confirm in ["yes", "y"]:
                            tc.delete_entity(partition_key=ent["PartitionKey"], row_key=ent["RowKey"])
                            entities.pop(num - 1)
                            edited_count += 1
                            print(f"✓ Entry #{num} deleted")
                except ValueError:
                    print("✗ Invalid input")
                    
            elif choice == "5":
                await _present_dependency_results_markdown(table_name, entities)
                
            else:
                print("✗ Invalid choice")
        
        return edited_count

    async def _present_infrastructure_results_markdown(table_name: str, entities: list) -> None:
        """Present infrastructure table results in markdown format."""
        print(f"\n{'='*80}")
        print(f"## Table: {table_name} (Infrastructure Details)")
        print(f"{'='*80}\n")
        
        print("| # | VM Hostname | IP Address | OS | vCPU | RAM | Environment | Function |")
        print("|---|-------------|------------|-----|------|-----|-------------|----------|")
        
        for idx, ent in enumerate(entities, 1):
            vm = (ent.get("VMHostname", "")[:25] or "-").replace("|", "\\|")
            ip = (ent.get("IPAddress", "") or "-").replace("|", "\\|")
            os = (ent.get("OperatingSystem", "")[:20] or "-").replace("|", "\\|")
            vcpu = (ent.get("vCPU", "") or "-").replace("|", "\\|")
            ram = (ent.get("RAM", "") or "-").replace("|", "\\|")
            env = (ent.get("ServerEnvironment", "") or "-").replace("|", "\\|")
            func = (ent.get("ServerFunction", "") or "-").replace("|", "\\|")
            
            print(f"| {idx} | {vm} | {ip} | {os} | {vcpu} | {ram} | {env} | {func} |")
        
        print(f"\n**Total Servers:** {len(entities)}")

    async def _interactive_edit_infrastructure(table_name: str, entities: list, tsc) -> int:
        """Allow user to edit infrastructure entries."""
        tc = tsc.get_table_client(table_name=table_name)
        edited_count = 0
        
        while True:
            print("\n" + "="*60)
            print("INFRASTRUCTURE VALIDATION OPTIONS:")
            print("="*60)
            print("1. Approve all entries")
            print("2. Edit specific server")
            print("3. View full server details")
            print("4. Re-display table")
            print("-"*60)
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                print("✓ All infrastructure entries approved")
                break
                
            elif choice == "2":
                try:
                    num = int(input("Enter server number to edit: ").strip())
                    if 1 <= num <= len(entities):
                        ent = entities[num - 1]
                        print(f"\n--- Editing Server #{num} ---")
                        print("Leave blank to keep current value")
                        
                        fields = [
                            ("VMHostname", "VM Hostname"),
                            ("Domain", "Domain"),
                            ("IPAddress", "IP Address"),
                            ("ServerFunction", "Server Function (Web/App/DB/Others)"),
                            ("OnpremSecurityZone", "Security Zone"),
                            ("OperatingSystem", "Operating System"),
                            ("vCPU", "vCPU"),
                            ("RAM", "RAM (GB)"),
                            ("DisksAndSize", "Disks and Size"),
                            ("ServerEnvironment", "Environment (Dev/Test/UAT/Prod)"),
                            ("GeneralNotes", "Notes")
                        ]
                        
                        updated = False
                        for field, label in fields:
                            current = ent.get(field, "")
                            new_val = input(f"{label} [{current}]: ").strip()
                            if new_val:
                                ent[field] = new_val
                                updated = True
                        
                        if updated:
                            ent["Confidence"] = 0.95
                            ent["Citation"] = "User validated/edited"
                            from azure.data.tables import UpdateMode as TableUpdateMode
                            tc.upsert_entity(entity=ent, mode=TableUpdateMode.MERGE)
                            edited_count += 1
                            print(f"✓ Server #{num} updated")
                except ValueError:
                    print("✗ Invalid input")
                    
            elif choice == "3":
                try:
                    num = int(input("Enter server number to view details: ").strip())
                    if 1 <= num <= len(entities):
                        ent = entities[num - 1]
                        print(f"\n--- Full Details for Server #{num} ---")
                        for key, value in ent.items():
                            if key not in ["PartitionKey", "RowKey", "etag", "Timestamp"]:
                                print(f"{key}: {value}")
                except ValueError:
                    print("✗ Invalid input")
                    
            elif choice == "4":
                await _present_infrastructure_results_markdown(table_name, entities)
                
            else:
                print("✗ Invalid choice")
        
        return edited_count

    # Modify the main workflow to use validation functions
    async def _process_all_qa_tables_with_validation(client_obj, agent_id: str) -> dict:
        """Process questions in all QA tables with validation after each table."""
        tables_to_process = [
            f"AppDetails{application_id}",
            f"PrivacyAndSecurity{application_id}",
            f"MSSQLDB{application_id}",
            f"OracleDB{application_id}"
        ]
        
        all_results = {}
        all_low_confidence = []
        
        for table_name in tables_to_process:
            # Check if table exists before processing
            try:
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    from azure.data.tables import TableServiceClient
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    from azure.data.tables import TableServiceClient
                    tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        continue
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                
                # Try to get table client to verify it exists
                tc = tsc.get_table_client(table_name=table_name)
                # Do a quick query to verify table exists
                try:
                    next(tc.query_entities(max_page_size=1), None)
                except Exception as e:
                    if "TableNotFound" in str(e) or "ResourceNotFound" in str(e):
                        logger.debug(f"Table {table_name} not found, skipping")
                        continue
                
                print(f"\n  Processing table: {table_name}...")
                # Use the new validation-enabled function
                result = await _process_questions_with_validation(client_obj, agent_id, table_name, application_id)
                all_results[table_name] = result
                
                if result.get("result") == "ok":
                    print(f"    ✓ Processed {result.get('answered')} questions")
                    if result.get('edited', 0) > 0:
                        print(f"    ✓ Edited {result.get('edited')} responses")
                    if result.get('lowConfidenceResolved', 0) > 0:
                        print(f"    ✓ Resolved {result.get('lowConfidenceResolved')} low confidence responses")
                    if result.get('lowConfidence', 0) > 0:
                        print(f"    ⚠ Remaining low confidence: {result.get('lowConfidence', 0)}")
                    
                    # Collect remaining low confidence questions
                    all_low_confidence.extend(result.get("lowConfidenceQuestions", []))
                    
            except Exception as ex:
                logger.warning(f"Failed to process table {table_name}: {ex}")
                continue
        
        return {
            "result": "ok",
            "tables_processed": len(all_results),
            "results": all_results,
            "total_low_confidence": len(all_low_confidence),
            "all_low_confidence_questions": all_low_confidence
        }

    async def _process_questions_for_table(client_obj, agent_id: str, table_name: str, partition_key: str) -> dict:
        """Process all questions for a specific table using single persistent thread."""
        try:
            from azure.data.tables import TableServiceClient, UpdateMode as TableUpdateMode
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            from semantic_kernel.agents import AzureAIAgent as SKAgent
        except Exception as ex:
            return {"result": "error", "message": f"Missing deps: {ex}"}

        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if conn_str:
            tsc = TableServiceClient.from_connection_string(conn_str)
        else:
            tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
            if not tables_url:
                return {"result": "unverified", "reason": "Missing AZURE_TABLES_ACCOUNT_URL"}
            cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
            tsc = TableServiceClient(endpoint=tables_url, credential=cred)

        tc = tsc.get_table_client(table_name=table_name)
        
        # Get all entities
        escaped_pk = str(partition_key).replace("'", "''")
        server_filter = f"PartitionKey eq '{escaped_pk}'"
        entities = list(tc.query_entities(query_filter=server_filter))
        
        # Get or create persistent thread for answer agent
        nonlocal answer_agent_thread_id
        
        if not answer_agent_thread_id:
            # Create a single thread for all Q&A processing
            thread = await client_obj.agents.threads.create()
            answer_agent_thread_id = thread.id
            agent_threads[agent_id] = answer_agent_thread_id
            logger.debug(f"Created persistent Q&A thread: {answer_agent_thread_id}")
        
        # Get agent
        try:
            agent_def = await client_obj.agents.get_agent(agent_id)
            answer_agent = SKAgent(client=client_obj, definition=agent_def)
        except Exception as ex:
            return {"result": "error", "message": f"Wrap agent failed: {ex}"}

        # Use the existing thread directly - no need for wrapper
        qa_thread = None  # Will use the thread in get_response calls

        answered = 0
        low_conf_count = 0
        low_confidence_questions = []
        
        for ent in entities:
            q = ent.get("Question")
            if not q:
                continue
            
            try:
                
                # Get response using persistent thread via direct API calls
                try:
                    # Add message to persistent thread
                    await client_obj.agents.messages.create(
                        thread_id=answer_agent_thread_id,
                        role="user",
                        content=q
                    )
                    
                    # Create run
                    run = await client_obj.agents.runs.create(
                        thread_id=answer_agent_thread_id,
                        agent_id=agent_id
                    )
                    
                    # Wait for completion
                    import asyncio
                    while run.status in ["queued", "in_progress", "requires_action"]:
                        await asyncio.sleep(1)
                        run = await client_obj.agents.runs.get(
                            thread_id=answer_agent_thread_id,
                            run_id=run.id
                        )
                    
                    if run.status == "completed":
                        # Get the assistant's response
                        messages = client_obj.agents.messages.list(thread_id=answer_agent_thread_id)
                        text = ""
                        async for message in messages:
                            if message.role == "assistant":
                                text = message.content[0].text.value if message.content else ""
                                break
                    else:
                        text = f"Run failed with status: {run.status}"
                        
                except Exception as run_ex:
                    logger.error(f"Direct API call failed for question: {run_ex}")
                    # Fallback to original method without persistent thread
                    resp = await answer_agent.get_response(messages=q, thread=None)
                    text = str(resp).strip()
                
                text = text.strip()
                logger.debug(f"Agent response for '{q[:50]}...': {text[:200]}")
                
                # Capture thread id
                try:
                    if getattr(resp, "thread", None) is not None:
                        last_answer_thread_id = getattr(resp.thread, "id", None) or last_answer_thread_id
                except Exception:
                    pass
                
                # Parse response
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    if "{" in text and "}" in text:
                        try:
                            json_part = text[text.find("{") : text.rfind("}") + 1]
                            parsed = json.loads(json_part)
                        except Exception:
                            parsed = None
                
                # Calculate confidence based on response quality and conflict detection
                confidence = 0.0
                conflict_detected = False
                conflicting_values = []
                
                if parsed and isinstance(parsed, dict):
                    response_text = parsed.get("Response") or parsed.get("response") or ""
                    citation = parsed.get("Citation") or parsed.get("citation") or ""
                    conflict_detected = parsed.get("ConflictDetected", False)
                    conflicting_values = parsed.get("ConflictingValues", [])
                    
                    # Clean up citation text
                    import re
                    citation = re.sub(r'【[^】]*】', '', citation)  # Remove special brackets and content
                    citation = citation.strip()
                    
                    # Get confidence from parsed response or calculate it
                    parsed_confidence = parsed.get("Confidence") or parsed.get("confidence")
                    if parsed_confidence is not None:
                        try:
                            confidence = float(parsed_confidence)
                        except (ValueError, TypeError):
                            confidence = 0.5
                    else:
                        # Determine confidence level based on response quality
                        if conflict_detected:
                            # Very low confidence for conflicts
                            confidence = 0.2
                        elif response_text and citation:
                            # Has both response and citation - high confidence
                            confidence = 0.9
                        elif response_text:
                            # Has response but no citation - medium confidence
                            confidence = 0.5
                        else:
                            # No response - low confidence
                            confidence = 0.0
                    
                    # Update entity
                    ent["Response"] = response_text
                    ent["Guidance"] = parsed.get("Guidance") or parsed.get("guidance") or ent.get("Guidance", "")
                    ent["Confidence"] = confidence
                    ent["Citation"] = citation
                    
                    # Add conflict information if detected
                    if conflict_detected:
                        ent["ConflictDetected"] = True
                        ent["ConflictingValues"] = json.dumps(conflicting_values)
                        # Add warning to response if not already present
                        if "CONFLICT" not in response_text.upper():
                            ent["Response"] = f"CONFLICT DETECTED: {response_text}"
                else:
                    # Fallback if JSON parsing failed
                    ent["Response"] = text
                    ent["Confidence"] = 0.3
                    ent["Citation"] = ""
                
                # Track low confidence questions (including conflicts)
                if ent["Confidence"] < 0.5 or conflict_detected:
                    low_confidence_questions.append({
                        "Table": table_name,
                        "RowKey": ent.get("RowKey"),
                        "Question": q,
                        "Response": ent.get("Response", ""),
                        "Confidence": ent.get("Confidence", 0.0),
                        "ConflictDetected": conflict_detected,
                        "ConflictingValues": conflicting_values
                    })
                    low_conf_count += 1
                
                # Update table
                tc.upsert_entity(entity=ent, mode=TableUpdateMode.MERGE)
                answered += 1
                
            except Exception as ex:
                logger.warning(f"Failed to process question for row {ent.get('RowKey')}: {ex}")
        
        return {
            "result": "ok",
            "table": table_name,
            "answered": answered,
            "total": len(entities),
            "lowConfidence": low_conf_count,
            "lowConfidenceQuestions": low_confidence_questions
        }

    async def _process_all_qa_tables(client_obj, agent_id: str) -> dict:
        """Process questions in all QA tables."""
        tables_to_process = [
            f"AppDetails{application_id}",
            f"PrivacyAndSecurity{application_id}",
            f"MSSQLDB{application_id}",       # Add new Q&A table
            f"OracleDB{application_id}"        # Add new Q&A table
        ]
        
        all_results = {}
        all_low_confidence = []
        
        for table_name in tables_to_process:
            # Check if table exists before processing
            try:
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    from azure.data.tables import TableServiceClient
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    from azure.data.tables import TableServiceClient
                    tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        continue
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                
                # Try to get table client to verify it exists
                tc = tsc.get_table_client(table_name=table_name)
                # Do a quick query to verify table exists
                try:
                    next(tc.query_entities(max_page_size=1), None)
                except Exception as e:
                    if "TableNotFound" in str(e) or "ResourceNotFound" in str(e):
                        logger.debug(f"Table {table_name} not found, skipping")
                        continue
                
                print(f"  Processing table: {table_name}...")
                result = await _process_questions_for_table(client_obj, agent_id, table_name, application_id)
                all_results[table_name] = result
                
                if result.get("result") == "ok":
                    print(f"    ✓ Processed {result.get('answered')} questions")
                    if result.get('lowConfidence', 0) > 0:
                        print(f"    ⚠ Low confidence: {result.get('lowConfidence', 0)}")
                    
                    # Collect all low confidence questions
                    all_low_confidence.extend(result.get("lowConfidenceQuestions", []))
                    
            except Exception as ex:
                logger.warning(f"Failed to process table {table_name}: {ex}")
                continue
        
        return {
            "result": "ok",
            "tables_processed": len(all_results),
            "results": all_results,
            "total_low_confidence": len(all_low_confidence),
            "all_low_confidence_questions": all_low_confidence
        }

    async def _interactive_low_confidence_resolution(low_confidence_questions: list) -> None:
        """Interactively resolve low confidence answers with user for multiple tables."""
        
        if not low_confidence_questions:
            print("No low confidence questions to resolve.")
            return
        
        # Group questions by table
        questions_by_table = {}
        for q in low_confidence_questions:
            table = q.get("Table", "Unknown")
            if table not in questions_by_table:
                questions_by_table[table] = []
            questions_by_table[table].append(q)
        
        # Separate conflicts from other low confidence
        conflict_count = sum(1 for q in low_confidence_questions if q.get("ConflictDetected", False))
        
        print(f"\n=== Resolving {len(low_confidence_questions)} Low Confidence Questions ===")
        if conflict_count > 0:
            print(f"⚠ WARNING: {conflict_count} questions have CONFLICTING answers in the search index")
        print(f"Tables with low confidence questions: {', '.join(questions_by_table.keys())}")
        print("Please provide better answers for questions where the system had low confidence.\n")
        
        try:
            from azure.data.tables import TableServiceClient, UpdateMode as TableUpdateMode
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if conn_str:
                tsc = TableServiceClient.from_connection_string(conn_str)
            else:
                tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                if not tables_url:
                    print("Cannot connect to table storage")
                    return
                cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                tsc = TableServiceClient(endpoint=tables_url, credential=cred)
            
            overall_idx = 0
            for table_name, questions in questions_by_table.items():
                print(f"\n--- Table: {table_name} ({len(questions)} questions) ---")
                tc = tsc.get_table_client(table_name=table_name)
                
                for idx, lc_q in enumerate(questions, 1):
                    overall_idx += 1
                    print(f"\nQuestion {overall_idx}/{len(low_confidence_questions)} (Table: {table_name}):")
                    print(f"Q: {lc_q['Question']}")
                    
                    # Special handling for conflicts
                    if lc_q.get("ConflictDetected", False):
                        print("⚠ CONFLICT DETECTED! Multiple values found in search index:")
                        conflicting_values = lc_q.get("ConflictingValues", [])
                        if conflicting_values:
                            for i, value in enumerate(conflicting_values, 1):
                                print(f"  {i}. {value}")
                        print(f"\nSystem Response: {lc_q['Response']}")
                        print("\nPlease provide the correct value or choose from the conflicting values above.")
                    else:
                        print(f"Current Answer (Confidence: {lc_q['Confidence']:.2f}): {lc_q['Response']}")
                    
                    try:
                        if lc_q.get("ConflictDetected", False):
                            user_answer = input("\nProvide the correct answer or enter a number to choose from conflicts above: ").strip()
                            
                            # Check if user entered a number to select from conflicts
                            if user_answer.isdigit() and conflicting_values:
                                choice = int(user_answer)
                                if 1 <= choice <= len(conflicting_values):
                                    user_answer = conflicting_values[choice - 1]
                                    print(f"Selected: {user_answer}")
                        else:
                            user_answer = input("\nProvide a better answer (or press Enter to keep current): ").strip()
                        
                        if user_answer:
                            # Update the table with user's answer
                            entity = tc.get_entity(partition_key=application_id, row_key=lc_q['RowKey'])
                            entity["Response"] = user_answer
                            entity["Confidence"] = 0.95  # High confidence for user-provided answers
                            entity["Citation"] = "User resolved conflict" if lc_q.get("ConflictDetected") else "User provided"
                            
                            # Clear conflict flags if they exist
                            if "ConflictDetected" in entity:
                                del entity["ConflictDetected"]
                            if "ConflictingValues" in entity:
                                del entity["ConflictingValues"]
                                
                            tc.upsert_entity(entity=entity, mode=TableUpdateMode.MERGE)
                            print("✓ Answer updated successfully")
                        else:
                            print("✓ Keeping current answer")
                        
                    except KeyboardInterrupt:
                        print("\n\nSkipping remaining questions...")
                        break
                    except Exception as ex:
                        print(f"Failed to update answer: {ex}")
                
                if overall_idx < len(low_confidence_questions):
                    try:
                        continue_prompt = input(f"\nContinue with next table? (yes/no): ").strip().lower()
                        if continue_prompt not in ["yes", "y"]:
                            break
                    except KeyboardInterrupt:
                        break
            
            print("\n=== Low confidence resolution complete ===\n")
            
        except Exception as ex:
            print(f"Error during low confidence resolution: {ex}")

    async def _export_app_tables_to_blob(app_id: str) -> dict:
        """Export application-specific Azure Table entities into a single JSONL blob for indexing.

        Returns dict with status, blob_url (if uploaded), records, tables_exported.
        """
        try:
            from azure.data.tables import TableServiceClient
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
            import io, json, datetime

            logger.debug(f"[export] Starting export for app_id={app_id}")
            # Core tables for the app. Allow override via env (comma-separated)
            env_tables = os.getenv("APP_TABLE_EXPORT_LIST")
            if env_tables:
                tables = [t.strip() for t in env_tables.split(",") if t.strip()]
                logger.debug(f"[export] Using tables from APP_TABLE_EXPORT_LIST env: {tables}")
            else:
                tables = [
                    f"AppDetails{app_id}",
                    f"PrivacyAndSecurity{app_id}",
                    f"MSSQLDB{app_id}",
                    f"OracleDB{app_id}",
                    f"IntegrationDependency{app_id}",
                    f"InfrastructureDetails{app_id}"
                ]
                logger.debug(f"[export] Using default table list: {tables}")

            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if conn_str:
                tsc = TableServiceClient.from_connection_string(conn_str)
                bsc = BlobServiceClient.from_connection_string(conn_str)
                logger.debug("[export] Initialized TableServiceClient & BlobServiceClient via connection string")
            else:
                tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL") or (tables_url.replace(".table.", ".blob.") if tables_url else None)
                if not tables_url or not account_url:
                    logger.debug("[export] Missing tables_url or account_url for AAD credential path")
                    return {"status": "error", "message": "Missing table/blob account URLs"}
                cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                bsc = BlobServiceClient(account_url=account_url, credential=cred)
                logger.debug(f"[export] Initialized clients via AAD tables_url={tables_url} account_url={account_url}")

            # Ensure container exists (already created earlier normally)
            container_name = app_id.lower()
            container_name = re.sub(r"[^a-z0-9-]", "-", container_name)
            logger.debug(f"Using blob container: {container_name}")
            try:
                blob_container = bsc.get_container_client(container_name)
                blob_container.get_container_properties()
            except Exception:
                try:
                    blob_container = bsc.create_container(container_name)
                except Exception:
                    pass
                blob_container = bsc.get_container_client(container_name)

            total_records = 0
            exported_tables = []
            per_table_counts = {}
            buf = io.StringIO()

            for table in tables:
                original_table = table
                logger.debug(f"[export] Processing table {original_table}")
                probe_error = None
                try:
                    tc = tsc.get_table_client(table_name=table)
                    # Quick existence probe: use list_entities (no filter) with an iterator
                    try:
                        ent_iter = tc.list_entities(results_per_page=1)
                        first = next(iter(ent_iter), None)
                        if first is not None:
                            logger.debug(f"[export] Table {original_table} exists (first entity present)")
                        else:
                            logger.debug(f"[export] Table {original_table} exists but returned no entities on probe")
                    except TypeError as te:
                        # Some SDK versions may not support results_per_page param for list_entities
                        logger.debug(f"[export] Probe TypeError for table {original_table}: {te}; retrying simplified probe")
                        try:
                            ent_iter = tc.list_entities()
                            first = next(iter(ent_iter), None)
                            if first is not None:
                                logger.debug(f"[export] Table {original_table} exists (simplified probe)" )
                            else:
                                logger.debug(f"[export] Table {original_table} exists but empty (simplified probe)")
                        except Exception as inner_probe_ex:
                            probe_error = inner_probe_ex
                            logger.debug(f"[export] Simplified probe failed for {original_table}: {inner_probe_ex}")
                    except Exception as generic_probe_ex:
                        probe_error = generic_probe_ex
                        logger.debug(f"[export] Probe failed for {original_table}: {generic_probe_ex}")
                except Exception as ex_access:
                    logger.debug(f"[export] Table {original_table} access error: {ex_access}")
                    per_table_counts[original_table] = {"exists": False, "exported": 0, "error": str(ex_access)}
                    continue

                entities = []
                escaped = str(app_id).replace("'", "''")
                # Primary attempt: partition filter
                try:
                    entities = list(tc.query_entities(query_filter=f"PartitionKey eq '{escaped}'"))
                    logger.debug(f"[export] Table {original_table} partition query returned {len(entities)} entities for PartitionKey={escaped}")
                except Exception as pe:
                    logger.debug(f"[export] Table {original_table} partition query error: {pe}")
                    entities = []

                # Fallback: full scan (paginated) then client filter if partition key differs (case sensitivity or naming differences)
                if not entities:
                    logger.debug(f"[export] Table {original_table} performing fallback scan")
                    try:
                        try:
                            scan_iter = tc.list_entities(results_per_page=200)
                        except TypeError:
                            # Older signature without results_per_page
                            scan_iter = tc.list_entities()
                        count_limit = 5000
                        temp = []
                        for ent in scan_iter:
                            temp.append(ent)
                            if len(temp) >= count_limit:
                                break
                        logger.debug(f"[export] Table {original_table} fallback scan collected {len(temp)} entities (pre-filter)")
                        lowered_app = app_id.lower()
                        filtered = [r for r in temp if str(r.get("PartitionKey", "")).lower() == lowered_app or lowered_app in str(r.get("PartitionKey", "")).lower()]
                        # If few overall, keep all; else prefer filtered subset
                        entities = filtered if filtered else (temp if len(temp) <= 200 else [])
                        logger.debug(f"[export] Table {original_table} post-filter entity count {len(entities)}")
                    except Exception as fallback_ex:
                        logger.debug(f"[export] Table {original_table} fallback scan error: {fallback_ex}")
                        if probe_error:
                            per_table_counts[original_table] = {"exists": probe_error is None, "exported": 0, "probe_error": str(probe_error), "fallback_error": str(fallback_ex)}
                        entities = []

                if not entities:
                    per_table_counts[original_table] = {"exists": True, "exported": 0, "note": "no entities after queries"}
                    logger.debug(f"[export] Table {original_table} no entities exported")
                    continue

                exported_tables.append(original_table)
                exported_count = 0
                for e in entities:
                    # Remove service metadata for cleaner indexing
                    e.pop("etag", None)
                    e.pop("Timestamp", None)
                    e["_SourceTable"] = original_table
                    pk = e.get("PartitionKey", "")
                    rk = e.get("RowKey", "")
                    e["Key"] = f"{pk}_{rk}" if pk else rk
                    buf.write(json.dumps(e, ensure_ascii=False) + "\n")
                    total_records += 1
                    exported_count += 1
                per_table_counts[original_table] = {"exists": True, "exported": exported_count}
                logger.debug(f"[export] Table {original_table} exported {exported_count} entities (cumulative total {total_records})")

            if total_records == 0:
                logger.debug("[export] No records exported from any table")
                return {"status": "empty", "message": "No table data found to export", "per_table": per_table_counts}

            ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            blob_name = f"tables_snapshot_{app_id}_{ts}.jsonl"
            data_bytes = buf.getvalue().encode("utf-8")
            blob_client = blob_container.get_blob_client(blob_name)
            logger.debug(f"[export] Uploading blob {blob_name} size={len(data_bytes)} bytes")
            blob_client.upload_blob(data_bytes, overwrite=True)
            blob_url = getattr(blob_client, "url", blob_name)
            logger.debug(f"[export] Export complete records={total_records} blob_url={blob_url}")
            return {
                "status": "ok",
                "blob_url": blob_url,
                "records": total_records,
                "tables_exported": exported_tables,
                "blob_name": blob_name,
                "per_table": per_table_counts
            }
        except Exception as ex:
            logger.error(f"Failed to export tables: {ex}", exc_info=True)
            return {"status": "error", "message": str(ex)}

    # Main chat loop logic
    async with DefaultAzureCredential(exclude_shared_token_cache_credential=True) as creds:
        async with AzureAIAgent.create_client(credential=creds, endpoint=endpoint) as client:
            # Load instructions
            instructions_text = (
                f"You are an orchestrator for Application ID: '{application_id}'. "
                "Help manage the Q&A process including table creation, indexing, and answer retrieval."
            )
            
            try:
                agent_definition = await client.agents.create_agent(
                    model=deployment,
                    name=orchestrator_name,
                    instructions=instructions_text,
                )
            except Exception as ex:
                print(f"Failed to create orchestrator agent: {ex}", flush=True)
                sys.exit(1)

            # Wrap with SK AzureAIAgent
            try:
                agent = AzureAIAgent(client=client, definition=agent_definition, plugins=[OrchestratorPlugin()])
            except Exception:
                agent = AzureAIAgent(client=client, definition=agent_definition)

            thread: Optional[AzureAIAgentThread] = None

            # --- New: Create initial thread and kickoff run so that Step 1 is invoked via agent tools ---
            kickoff_run_completed = False
            kickoff_thread_id: Optional[str] = None
            try:
                kickoff_thread = await client.agents.threads.create()
                kickoff_thread_id = kickoff_thread.id
                # Store for reuse by answer agent / other agents (user request: reuse same thread)
                agent_threads[agent_definition.id] = kickoff_thread_id
                # Also pre-assign this as the persistent answer agent thread id so downstream processing reuses it
                answer_agent_thread_id = kickoff_thread_id
                # Provide initial system/user prompt instructing orchestrator to execute first step only then stop.
                initial_prompt = (
                    "You are the orchestrator agent for application ID '" + application_id + "'. "
                    "Immediately perform STEP 1 of the workflow ONLY: call the tool/function clone_all_templates to clone all required template tables. "
                    "After the tool completes, summarize the result briefly (one line) and DO NOT proceed to any other steps or actions. "
                    "End the run after reporting the summary so that other specialized agents can continue using this same thread."
                )
                await client.agents.messages.create(
                    thread_id=kickoff_thread_id,
                    role="user",
                    content=initial_prompt
                )
                kickoff_run = await client.agents.runs.create(
                    thread_id=kickoff_thread_id,
                    agent_id=agent_definition.id
                )
                # Poll for completion
                import asyncio as _asyncio_poll
                _poll_wait = 0
                while kickoff_run.status in ["queued", "in_progress", "requires_action"] and _poll_wait < 60:
                    await _asyncio_poll.sleep(1)
                    _poll_wait += 1
                    kickoff_run = await client.agents.runs.get(thread_id=kickoff_thread_id, run_id=kickoff_run.id)
                if kickoff_run.status == "completed":
                    kickoff_run_completed = True
                    # Retrieve last assistant message for logging
                    try:
                        msgs = client.agents.messages.list(thread_id=kickoff_thread_id)
                        async for m in msgs:
                            if m.role == "assistant":
                                txt = m.content[0].text.value if m.content else ""
                                logger.debug(f"Kickoff assistant message: {txt[:300]}")
                                break
                    except Exception:
                        pass
                else:
                    logger.warning(f"Kickoff run did not complete successfully (status={kickoff_run.status}). Will fallback to manual Step 1.")
                # Prepare AzureAIAgentThread object for future chat reuse
                if kickoff_thread_id:
                    thread = AzureAIAgentThread(client=client, thread_id=kickoff_thread_id)
            except Exception as kickoff_ex:
                logger.warning(f"Failed to perform kickoff run: {kickoff_ex}")
            
            # Startup sequence
            print(f"\n=== Starting Application Setup for '{application_id}' ===\n")
            
            # Step 1: Clone all template tables
            plugin = OrchestratorPlugin()
            if kickoff_run_completed:
                # We still query to show a user-facing summary of table states using direct tool (idempotent)
                try:
                    clone_result = plugin.clone_all_templates()
                    clone_data = json.loads(clone_result)
                    if clone_data.get("result") == "ok":
                        print("✓ Tables verified/created:")
                        for template, result in clone_data.get("cloned", {}).items():
                            if result.get("status") == "created":
                                print(f"  - {result.get('table')}: {result.get('copied')} questions copied")
                            elif result.get("status") == "exists":
                                print(f"  - {result.get('table')}: Already exists")
                    else:
                        print(f"⚠ Verification after kickoff run reported: {clone_data}")
                except Exception as _post_verify_ex:
                    print(f"⚠ Could not verify tables after kickoff run: {_post_verify_ex}")
            else:
                print("Step 1: Creating application-specific tables (fallback manual invocation)...")
                clone_result = plugin.clone_all_templates()
                clone_data = json.loads(clone_result)
                if clone_data.get("result") == "ok":
                    print("✓ Tables created:")
                    for template, result in clone_data.get("cloned", {}).items():
                        if result.get("status") == "created":
                            print(f"  - {result.get('table')}: {result.get('copied')} questions copied")
                        elif result.get("status") == "exists":
                            print(f"  - {result.get('table')}: Already exists")
                else:
                    print(f"✗ Failed to create tables: {clone_data}")
            
            # Step 2: Create container
            print("\nStep 2: Ensuring blob container exists...")
            container_result = plugin.check_container_exists()
            container_data = json.loads(container_result)
            if container_data.get("created"):
                print(f"✓ Container created: {container_data.get('url')}")
                print(f"   {container_data.get('message')}")
            elif container_data.get("exists"):
                print(f"✓ Container exists: {container_data.get('url')}")
            
            # Step 3: Ask user to upload files
            print("\nStep 3: File Upload")
            print(f"Please upload your documents to the container: {application_id}")
            upload_confirmed = input("Have you uploaded the files? (yes/no): ").strip().lower()
            
            # Step 4: Trigger indexing
            if upload_confirmed in ["yes", "y"]:
                print("\nStep 4: Triggering indexing function...")
                indexing_success = await _trigger_and_check_indexing(application_id)
                
                if not indexing_success:
                    print("✗ No documents were indexed. Please ensure files are uploaded.")
                    print("After uploading, use the /reindex command to retry.")
                else:
                    print("✓ Indexing completed successfully")
                    
                    # Step 5: Create agent and process questions
                    print("\nStep 5: Creating answer agent and processing questions...")
                    agent_id = await ensure_agent(application_id)
                    answer_agent_ids.add(agent_id)
                    if agent_id:
                        print(f"✓ Answer agent created: {agent_id}")
                        
                        print("\nProcessing questions from all QA tables...")
                        # Use the new validation-enabled function
                        qa_result = await _process_all_qa_tables_with_validation(client, agent_id)
                        
                        if qa_result.get("result") == "ok":
                            # Results already printed by the function
                            
                            # Collect all remaining low confidence questions
                            all_low_conf = qa_result.get("all_low_confidence_questions", [])
                            
                            # Step 6: Process dependency information with validation
                            print("\nStep 6: Extracting and populating dependency information...")
                            dep_result = await plugin.populate_dependency_table()
                            dep_data = json.loads(dep_result)
                            if dep_data.get("result") == "ok":
                                print(f"✓ Populated {dep_data.get('populated')} dependency records")
                                print(f"  Processed {dep_data.get('servers_processed')} unique servers")
                                # Ensure TableServiceClient initialized for validation steps
                                if 'tsc' not in locals() or tsc is None:
                                    try:
                                        from azure.data.tables import TableServiceClient
                                        from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                                        conn_str_val = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                                        if conn_str_val:
                                            tsc = TableServiceClient.from_connection_string(conn_str_val)
                                        else:
                                            tables_url_val = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                                            if tables_url_val:
                                                cred_val = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                                                tsc = TableServiceClient(endpoint=tables_url_val, credential=cred_val)
                                            else:
                                                tsc = None
                                    except Exception as _init_ex:
                                        logger.warning(f"Unable to initialize TableServiceClient for dependency validation: {_init_ex}")
                                        tsc = None

                                # Validate dependency entries
                                dep_table = f"IntegrationDependency{application_id}"
                                try:
                                    if tsc:
                                        tc = tsc.get_table_client(table_name=dep_table)
                                        dep_entities = list(tc.query_entities(query_filter=f"PartitionKey eq '{application_id}'"))
                                        if dep_entities:
                                            await _present_dependency_results_markdown(dep_table, dep_entities)
                                            validate_choice = input("Would you like to validate/edit these dependencies? (yes/no): ").strip().lower()
                                            if validate_choice in ["yes", "y"]:
                                                edited = await _interactive_edit_dependencies(dep_table, dep_entities, tsc)
                                                if edited > 0:
                                                    print(f"✓ Updated {edited} dependency entries")
                                    else:
                                        logger.warning("Skipping dependency validation: TableServiceClient unavailable")
                                except Exception as ex:
                                    logger.warning(f"Dependency validation failed: {ex}")
                            else:
                                print(f"⚠ Dependency extraction: {dep_data.get('message')}")
                            
                            # Step 7: Process infrastructure information with validation
                            print("\nStep 7: Extracting and populating infrastructure details...")
                            infra_result = await plugin.populate_infrastructure_table()
                            infra_data = json.loads(infra_result)
                            if infra_data.get("result") == "ok":
                                print(f"✓ Populated {infra_data.get('populated')} infrastructure records")
                                print(f"  Processed {infra_data.get('servers_processed')} servers")
                                # Ensure TableServiceClient initialized for infrastructure validation
                                if 'tsc' not in locals() or tsc is None:
                                    try:
                                        from azure.data.tables import TableServiceClient
                                        from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                                        conn_str_val = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                                        if conn_str_val:
                                            tsc = TableServiceClient.from_connection_string(conn_str_val)
                                        else:
                                            tables_url_val = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                                            if tables_url_val:
                                                cred_val = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                                                tsc = TableServiceClient(endpoint=tables_url_val, credential=cred_val)
                                            else:
                                                tsc = None
                                    except Exception as _init_ex:
                                        logger.warning(f"Unable to initialize TableServiceClient for infrastructure validation: {_init_ex}")
                                        tsc = None

                                # Validate infrastructure entries
                                infra_table = f"InfrastructureDetails{application_id}"
                                try:
                                    if tsc:
                                        tc = tsc.get_table_client(table_name=infra_table)
                                        infra_entities = list(tc.query_entities(query_filter=f"PartitionKey eq '{application_id}'"))
                                        if infra_entities:
                                            await _present_infrastructure_results_markdown(infra_table, infra_entities)
                                            validate_choice = input("Would you like to validate/edit these infrastructure details? (yes/no): ").strip().lower()
                                            if validate_choice in ["yes", "y"]:
                                                edited = await _interactive_edit_infrastructure(infra_table, infra_entities, tsc)
                                                if edited > 0:
                                                    print(f"✓ Updated {edited} infrastructure entries")
                                    else:
                                        logger.warning("Skipping infrastructure validation: TableServiceClient unavailable")
                                except Exception as ex:
                                    logger.warning(f"Infrastructure validation failed: {ex}")
                            else:
                                print(f"⚠ Infrastructure extraction: {infra_data.get('message')}")
                        
                        # Step 8: Final check for any remaining low confidence questions
                        if len(all_low_conf) > 0:
                            print(f"\n⚠ {len(all_low_conf)} low confidence questions remain after validation")
                            resolve = input("Would you like to resolve remaining low confidence questions? (yes/no): ").strip().lower()
                            if resolve in ["yes", "y"]:
                                await _interactive_low_confidence_resolution(all_low_conf)
                        else:
                            print("\n✓ All questions have been processed with acceptable confidence levels")
            
            print(f"\n=== Setup Complete ===")
            print(f"Orchestrator ready. Type 'help' for available commands or 'exit' to quit.\n")
            # Intermediate Step: Index consolidated table content
            print("\nIndex application tables into search index")
            proceed_index = input("Proceed to export & reindex table data into search? (yes/no): ").strip().lower()
            if proceed_index in ["yes", "y"]:
                print("  - Exporting tables to blob snapshot...")
                export_info = await _export_app_tables_to_blob(application_id)
                if export_info.get("status") == "ok":
                    print(f"  ✓ Exported {export_info['records']} records from {len(export_info['tables_exported'])} tables")
                    print(f"    Blob: {export_info.get('blob_name')}")
                    per = export_info.get("per_table", {})
                    if per:
                        print("    Per-table counts:")
                        for t, meta in per.items():
                            print(f"      - {t}: exported={meta.get('exported')} exists={meta.get('exists')}")
                elif export_info.get("status") == "empty":
                    print("  ⚠ No table data found to export. Continuing.")
                    per = export_info.get("per_table", {})
                    if per:
                        print("    Diagnostics:")
                        for t, meta in per.items():
                            print(f"      - {t}: exported={meta.get('exported')} exists={meta.get('exists')}")
                    print("    Hints: Verify tables have PartitionKey='" + application_id + "' or set APP_TABLE_EXPORT_LIST if custom names.")
                else:
                    print(f"  ⚠ Export failed: {export_info.get('message')}")
                print("  - Triggering search indexing function to incorporate latest table data...")
                idx_ok = await _trigger_and_check_indexing(application_id)
                if idx_ok:
                    print("  ✓ Reindex triggered/completed successfully with table snapshot")
                else:
                    print("  ⚠ Reindex did not confirm document ingestion")
            else:
                print("  Skipping intermediate reindex step at user request.")

            # Run ASR Agent after intermediate indexing
            print(f"\n=== Running ASR Agent ===")
            print(f"Invoking ASR Agent for Application ID: {application_id}")
            try:
                thread_id = agent_threads.get(agent_id) or answer_agent_thread_id
                asr_thread = None
                if thread_id:
                    asr_thread = AzureAIAgentThread(client=client, thread_id=thread_id)
                    print(f"Using existing thread ID: {thread_id}")
                else:
                    print("No existing thread found, ASR agent will create a new one")
                asr_result = await run_asr_agent(application_id, client, asr_thread)
                if asr_result.get("status") == "success":
                    print(f"✓ ASR Agent execution completed successfully")
                    print(f"  - Agent ID: {asr_result.get('agent_id')}")
                    print(f"  - Output file: {asr_result.get('output_file')}")
                    print(f"  - Markdown file: {asr_result.get('markdown_file')}")
                    print(f"  - Blob URL: {asr_result.get('blob_url')}")
                    
                    # Store ASR agent ID for cleanup
                    if asr_result.get('agent_id'):
                        answer_agent_ids.add(asr_result.get('agent_id'))
                    if asr_result.get('thread'):
                        thread = asr_result.get('thread')
                else:
                    print(f"⚠ ASR Agent execution failed: {asr_result.get('message', 'Unknown error')}")
            except Exception as ex:
                print(f"⚠ Error running ASR Agent: {str(ex)}")
            print(f"=== ASR Agent Complete ===\n")
            
            # Interactive loop
            try:
                while True:
                    try:
                        user_input = input("You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    
                    if user_input.lower() in {"exit", "quit"}:
                        break
                    
                    # Help command
                    if user_input.lower() == "help":
                        print("\nAvailable commands:")
                        print("  /reindex - Retry indexing after uploading files")
                        print("  /processqa - Process all questions again")
                        print("  /processinfra - Process infrastructure details")
                        print("  /resolvelow - Resolve low confidence questions")
                        print("  /status - Show current status")
                        print("  /qasummary - Show Q&A summary")
                        print("  /deleteindex - Delete the search index for this application")
                        print("  help - Show this help")
                        print("  exit - Quit the application")
                        print("\nNote: ASR (Application Summary Report) agent has already been executed during setup.\n")
                        continue
                    
                    # Reindex command
                    if user_input.startswith("/reindex"):
                        print("Triggering reindexing...")
                        success = await _trigger_and_check_indexing(application_id)
                        if success:
                            print("✓ Reindexing successful")
                        else:
                            print("✗ Reindexing failed or no documents found")
                        continue
                    
                    # Process QA command
                    if user_input.startswith("/processqa"):
                        agent_id = await ensure_agent(application_id)
                        answer_agent_ids.add(agent_id)
                        if agent_id:
                            result = await _process_all_qa_tables(client, agent_id)
                            if result.get("result") == "ok":
                                for table_name, table_result in result.get("results", {}).items():
                                    print(f"{table_name}: {table_result.get('answered')} processed, {table_result.get('lowConfidence')} low confidence")
                                
                                # Ask if user wants to resolve low confidence questions
                                all_low_conf = result.get("all_low_confidence_questions", [])
                                if len(all_low_conf) > 0:
                                    resolve = input(f"\n{len(all_low_conf)} low confidence questions found. Resolve now? (yes/no): ").strip().lower()
                                    if resolve in ["yes", "y"]:
                                        await _interactive_low_confidence_resolution(all_low_conf)
                        else:
                            print("Failed to create/get agent")
                        continue
                    
                    # Resolve low confidence
                    if user_input.startswith("/resolvelow"):
                        # Get current low confidence questions from all tables
                        tables = [f"AppDetails{application_id}", f"PrivacyAndSecurity{application_id}"]
                        current_low_conf = []
                        # Establish table client
                        try:
                            from azure.data.tables import TableServiceClient
                            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                            if conn_str:
                                tsc = TableServiceClient.from_connection_string(conn_str)
                            else:
                                tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                                if tables_url:
                                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                                else:
                                    tsc = None
                        except Exception:
                            tsc = None
                        
                        for table_name in tables:
                            try:
                                if not tsc:
                                    continue
                                tc = tsc.get_table_client(table_name=table_name)
                                entities = list(tc.query_entities(query_filter=f"PartitionKey eq '{application_id}'"))
                                for ent in entities:
                                    if ent.get("Confidence", 1.0) < 0.5 and ent.get("Question"):
                                        current_low_conf.append({
                                            "Table": table_name,
                                            "RowKey": ent.get("RowKey"),
                                            "Question": ent.get("Question"),
                                            "Response": ent.get("Response", ""),
                                            "Confidence": ent.get("Confidence", 0.0)
                                        })
                            except:
                                pass
                        
                        if current_low_conf:
                            await _interactive_low_confidence_resolution(current_low_conf)
                        else:
                            print("No low confidence questions found.")
                        continue
                    
                    # Status command
                    if user_input.startswith("/status"):
                        status_result = plugin.get_indexing_status()
                        print(f"Indexing status: {status_result}")
                        continue
                    
                    # QA Summary command
                    if user_input.startswith("/qasummary"):
                        summary_result = plugin.get_qa_summary()
                        print(f"QA summary: {summary_result}")
                        continue
                    
                    # Delete Index command
                    if user_input.startswith("/deleteindex"):
                        try:
                            index_name = _sanitize_index_name(application_id)
                            # Check if index exists
                            index_exists = check_index_exists(index_name)
                            
                            if index_exists:
                                confirm = input(f"Are you sure you want to delete the search index '{index_name}'? (yes/no): ").strip().lower()
                                if confirm in {"yes", "y"}:
                                    if delete_search_index(application_id):
                                        print(f"✓ Successfully deleted search index: {index_name}")
                                    else:
                                        print(f"✗ Failed to delete search index: {index_name}")
                                else:
                                    print("Index deletion cancelled.")
                            elif index_exists is False:
                                print(f"Search index '{index_name}' does not exist.")
                            else:
                                print(f"Could not verify if search index '{index_name}' exists.")
                        except Exception as ex:
                            print(f"Error during index deletion: {ex}")
                        continue
                    
                    # Normal chat
                    try:
                        response = await agent.get_response(messages=user_input, thread=thread)
                        print(f"Assistant: {response}")
                        thread = response.thread
                    except Exception as ex:
                        print(f"Assistant error: {ex}")
                        
            finally:
                # Cleanup resources - simplified thread deletion
                if cleanup and os.getenv("APP_CLEANUP_ON_EXIT", "true").lower() in {"1", "true", "yes", "on"}:
                    print("\nInitiating cleanup of threads and agents ...")
                    
                    # Ask user for confirmation before deleting the search index
                    delete_index_confirmed = False
                    try:
                        index_name = _sanitize_index_name(application_id)
                        # Check if index exists before asking for confirmation
                        index_exists = check_index_exists(index_name)
                        
                        if index_exists:
                            delete_index_confirmed = True
                        elif index_exists is False:
                            print(f"  ℹ Search index '{index_name}' does not exist or was already deleted")
                        else:
                            print(f"  ⚠ Could not verify if search index '{index_name}' exists")
                    except KeyboardInterrupt:
                        print("\n  Skipping index deletion confirmation...")
                        delete_index_confirmed = False
                    except Exception as ex:
                        logger.warning(f"Error during index deletion confirmation: {ex}")
                        delete_index_confirmed = False
                    
                    # Delete search index if confirmed
                    if delete_index_confirmed:
                        print(f"  Deleting search index '{index_name}'...")
                        if delete_search_index(application_id):
                            print(f"  ✓ Deleted search index: {index_name}")
                        else:
                            print(f"  ⚠ Failed to delete search index: {index_name}")
                    
                    # Delete main orchestrator thread
                    try:
                        if thread and getattr(thread, "id", None):
                            await client.agents.threads.delete(thread_id=thread.id)
                            print(f"  ✓ Deleted main orchestrator thread: {thread.id}")
                    except Exception as ex:
                        print(f"  ⚠ Could not delete main thread: {ex}")
                    
                    # Delete all persistent agent threads
                    deleted_threads = 0
                    for agent_id, thread_id in agent_threads.items():
                        try:
                            await client.agents.threads.delete(thread_id=thread_id)
                            deleted_threads += 1
                            logger.debug(f"Deleted thread {thread_id} for agent {agent_id}")
                        except Exception:
                            pass
                    if deleted_threads:
                        print(f"  ✓ Deleted {deleted_threads} agent threads")

                    # Delete ephemeral threads created during dependency / infrastructure extraction
                    ephemeral_deleted = 0
                    for t_id in list(ephemeral_thread_ids):
                        try:
                            await client.agents.threads.delete(thread_id=t_id)
                            ephemeral_deleted += 1
                        except Exception:
                            pass
                    if ephemeral_deleted:
                        print(f"  ✓ Deleted {ephemeral_deleted} ephemeral threads")

                    # Delete answer agents (intake agents) gathered during session
                    deleted_agents = 0
                    for a_id in list(answer_agent_ids):
                        try:
                            await client.agents.delete_agent(agent_id=a_id)
                            deleted_agents += 1
                            print(f"  ✓ Deleted answer agent: {a_id}")
                        except Exception as ex:
                            print(f"  ⚠ Could not delete answer agent {a_id}: {ex}")

                    # Delete orchestrator agent
                    try:
                        if agent_definition and getattr(agent_definition, "id", None):
                            await client.agents.delete_agent(agent_id=agent_definition.id)
                            print(f"  ✓ Deleted orchestrator agent: {agent_definition.id}")
                    except Exception as ex:
                        print(f"  ⚠ Could not delete orchestrator agent: {ex}")

                    print("Cleanup complete.\n")
                else:
                    try:
                        await thread.delete() if thread else None
                    except Exception:
                        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Orchestrator Agent")
    parser.add_argument(
        "--orchestrator-name",
        default=os.environ.get("ORCHESTRATOR_AGENT_NAME", "orchestrator"),
        help="Orchestrator agent name (default: 'orchestrator')",
    )
    parser.add_argument(
        "--application-id",
        help="Application ID to use",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable automatic deletion of agents and threads on exit"
    )
    args = parser.parse_args()
    
    import asyncio as _asyncio
    cleanup_enabled = not args.no_cleanup
    _asyncio.run(chat_loop(args.orchestrator_name, args.application_id, cleanup=cleanup_enabled))


if __name__ == "__main__":
    main()
