from typing import Optional, List, Dict, Any

import argparse
import json
import logging
import os
import sys
import aiohttp
import asyncio

from dotenv import load_dotenv

from logging_config import configure_logging
from intake_agent import ensure_agent


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


async def trigger_indexing_function(app_id: str, container_name: str) -> Dict[str, Any]:
    """Trigger the indexing function app and return the result."""
    function_url = os.getenv("AZURE_INDEXING_FUNCTION_URL")
    function_key = os.getenv("AZURE_INDEXING_FUNCTION_KEY")
    
    if not function_url:
        raise ValueError("AZURE_INDEXING_FUNCTION_URL not set")
    
    headers = {
        "Content-Type": "application/json"
    }
    if function_key:
        headers["x-functions-key"] = function_key
    
    payload = {
        "appId": app_id,
        "container": container_name
    }
    
    logger.debug(f"Triggering indexing function for appId={app_id}, container={container_name}")
    logger.debug(f"Function URL: {function_url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(function_url, json=payload, headers=headers) as response:
            result = await response.json()
            logger.debug(f"Indexing function response: {result}")
            return result


async def chat_loop(orchestrator_name: str, application_id: Optional[str] = None) -> None:
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

    # Track the most recent answer-agent thread id for diagnostics
    last_answer_thread_id: Optional[str] = None
    # Track low confidence questions for interactive resolution
    low_confidence_questions: List[Dict[str, Any]] = []

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
                if not container_name:
                    container_name = application_id
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    try:
                        from azure.storage.blob import BlobServiceClient
                        bsc = BlobServiceClient.from_connection_string(conn_str)
                    except Exception as ex:
                        return json.dumps({"result": "error", "message": f"BlobServiceClient init failed: {ex}"})
                else:
                    try:
                        from azure.storage.blob import BlobServiceClient
                        from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                        acct_url = account_url or os.getenv("AZURE_STORAGE_ACCOUNT_URL")
                        if not acct_url:
                            return json.dumps({"result": "unverified", "reason": "Missing AZURE_STORAGE_ACCOUNT_URL"})
                        cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                        bsc = BlobServiceClient(account_url=acct_url, credential=cred)
                    except Exception as ex:
                        return json.dumps({"result": "error", "message": f"BlobServiceClient init failed: {ex}"})

                try:
                    cc = bsc.get_container_client(container_name)
                    try:
                        cc.get_container_properties()
                        return json.dumps({
                            "container": container_name,
                            "exists": True,
                            "created": False,
                            "url": getattr(cc, "url", None),
                        })
                    except Exception as ex:
                        # Create the container if it does not exist
                        if ex.__class__.__name__ in {"ResourceNotFoundError", "ResourceNotFound"}:
                            try:
                                logger.debug(f"Creating container '{container_name}'")
                                cc.create_container()
                                return json.dumps({
                                    "container": container_name,
                                    "exists": True,
                                    "created": True,
                                    "url": getattr(cc, "url", None),
                                    "message": f"Container '{container_name}' created. Please upload your files to this container to proceed.",
                                })
                            except Exception as create_ex:
                                # If already created by race, treat as exists
                                if "ContainerAlreadyExists" in str(create_ex):
                                    return json.dumps({
                                        "container": container_name,
                                        "exists": True,
                                        "created": False,
                                        "url": getattr(cc, "url", None),
                                    })
                                return json.dumps({"result": "error", "message": f"Create container failed: {create_ex}"})
                        # Other errors
                        return json.dumps({"result": "unverified", "message": str(ex)})
                except Exception as ex:
                    return json.dumps({"result": "error", "message": str(ex)})
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        @kernel_function(description="Clone template table to create app-specific QA table. Copies all questions from template.")
        def clone_template_table(self, template_table: Optional[str] = None, target_table: Optional[str] = None) -> str:
            """Clone template table with questions to create app-specific table."""
            try:
                template = template_table or os.getenv("AZURE_QA_TEMPLATE_TABLE", "AppDetailsTemplate")
                target = target_table or f"AppDetails{application_id}"
                
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                try:
                    from azure.data.tables import TableServiceClient
                except Exception as ex:
                    return json.dumps({"result": "error", "message": f"TableServiceClient import failed: {ex}"})
                
                if conn_str:
                    tsc = TableServiceClient.from_connection_string(conn_str)
                else:
                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                    tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
                    if not tables_url:
                        return json.dumps({"result": "error", "reason": "Missing AZURE_TABLES_ACCOUNT_URL"})
                    cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                    tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                
                # Check if target table already exists
                table_exists = False
                try:
                    existing_tables = [t.name for t in tsc.list_tables()]
                    if target in existing_tables:
                        table_exists = True
                except Exception as ex:
                    logger.warning(f"Failed to list tables: {ex}")
                    # Try alternative check
                    try:
                        tc_target = tsc.get_table_client(table_name=target)
                        # Try a simple operation to check existence
                        try:
                            # Query with no results expected - if table doesn't exist, this will raise an error
                            next(tc_target.query_entities(max_page_size=1), None)
                            table_exists = True
                        except StopIteration:
                            # Empty table but exists
                            table_exists = True
                    except Exception as inner_ex:
                        if "TableNotFound" in str(inner_ex) or "ResourceNotFound" in str(inner_ex) or "404" in str(inner_ex):
                            table_exists = False
                        else:
                            logger.debug(f"Could not determine table existence: {inner_ex}")
                            table_exists = False
                
                if table_exists:
                    # Check if table has data already
                    tc_target = tsc.get_table_client(table_name=target)
                    try:
                        existing_count = sum(1 for _ in tc_target.query_entities(max_page_size=1000))
                        if existing_count > 0:
                            return json.dumps({
                                "result": "exists", 
                                "table": target, 
                                "message": f"Table already exists with {existing_count} rows"
                            })
                    except Exception:
                        pass
                    return json.dumps({"result": "exists", "table": target, "message": "Table already exists"})
                
                # Create the target table
                try:
                    tsc.create_table(table_name=target)
                    logger.debug(f"Created table: {target}")
                except Exception as e:
                    if "TableAlreadyExists" in str(e) or "AlreadyExists" in str(e):
                        logger.debug(f"Table {target} already exists (race condition)")
                    else:
                        return json.dumps({"result": "error", "message": f"Create table failed: {e}"})
                
                # Get table clients
                tc_template = tsc.get_table_client(table_name=template)
                tc_target = tsc.get_table_client(table_name=target)
                
                copied = 0
                try:
                    # Query all entities from template - use list() to get all entities
                    logger.debug(f"Querying entities from template table: {template}")
                    
                    # Use list() to query all entities from the template
                    template_entities = list(tc_template.list_entities())
                    
                    for entity in template_entities:
                        # Create new entity with updated PartitionKey
                        new_entity = {
                            "PartitionKey": application_id,
                            "RowKey": entity.get("RowKey", f"Q{copied+1:03d}"),
                            "Question": entity.get("Question", ""),
                            "Guidance": entity.get("Guidance", ""),
                            "Response": "",  # Empty initially
                            "Confidence": 0.0,  # Default to 0
                            "Citation": ""  # Empty initially
                        }
                        
                        # Only copy if Question field has content
                        if new_entity["Question"]:
                            tc_target.upsert_entity(entity=new_entity)
                            copied += 1
                    
                    if copied == 0:
                        logger.warning(f"No entities found in template table: {template}")
                        return json.dumps({
                            "result": "warning",
                            "template": template,
                            "target": target,
                            "copied": 0,
                            "message": f"Created table '{target}' but template '{template}' had no questions to copy"
                        })
                    
                    return json.dumps({
                        "result": "ok",
                        "template": template,
                        "target": target,
                        "copied": copied,
                        "message": f"Created table '{target}' with {copied} questions from template"
                    })
                    
                except Exception as ex:
                    error_msg = str(ex)
                    if "TableNotFound" in error_msg or "ResourceNotFound" in error_msg:
                        return json.dumps({
                            "result": "error", 
                            "message": f"Template table '{template}' not found. Please ensure it exists with questions."
                        })
                    return json.dumps({"result": "error", "message": f"Copy failed: {ex}"})
                    
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        @kernel_function(description="Clone multiple template tables for application setup.")
        def clone_all_templates(self, templates: Optional[List[str]] = None) -> str:
            """Clone all required template tables for the application."""
            try:
                default_templates = [
                    "AppDetailsTemplate",
                    "PrivacyAndSecurity", 
                    "IntegrationDependencyTemplate"
                ]
                templates_to_clone = templates or default_templates
                
                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                try:
                    from azure.data.tables import TableServiceClient
                except Exception as ex:
                    return json.dumps({"result": "error", "message": f"TableServiceClient import failed: {ex}"})
                
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
                    # Determine target table name based on template
                    if "AppDetails" in template:
                        target = f"AppDetails{application_id}"
                    elif "PrivacyAndSecurity" in template:
                        target = f"PrivacyAndSecurity{application_id}"
                    elif "IntegrationDependency" in template:
                        target = f"IntegrationDependency{application_id}"
                    else:
                        target = f"{template.replace('Template', '')}{application_id}"
                    
                    # Clone the template
                    result = self._clone_single_template(tsc, template, target, application_id)
                    results[template] = result
                
                return json.dumps({"result": "ok", "cloned": results})
                
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})

        def _clone_single_template(self, tsc, template: str, target: str, app_id: str) -> dict:
            """Helper to clone a single template table."""
            try:
                # Check if target exists
                try:
                    tc_target = tsc.get_table_client(table_name=target)
                    existing_count = sum(1 for _ in tc_target.query_entities(max_page_size=1))
                    if existing_count > 0:
                        return {"status": "exists", "table": target, "rows": existing_count}
                except:
                    pass
                
                # Create target table
                try:
                    tsc.create_table(table_name=target)
                except Exception as e:
                    if "TableAlreadyExists" not in str(e):
                        logger.warning(f"Create table warning: {e}")
                
                # Copy entities
                tc_template = tsc.get_table_client(table_name=template)
                tc_target = tsc.get_table_client(table_name=target)
                
                copied = 0
                for entity in tc_template.list_entities():
                    # Update PartitionKey to application_id
                    entity["PartitionKey"] = app_id
                    # Ensure required fields exist
                    if "Response" not in entity:
                        entity["Response"] = ""
                    if "Confidence" not in entity:
                        entity["Confidence"] = 0.0
                    if "Citation" not in entity:
                        entity["Citation"] = ""
                    tc_target.upsert_entity(entity=entity)
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
                        # Create a thread for this query
                        thread = await ai_client.agents.threads.create()
                        
                        # Send the query
                        await ai_client.agents.messages.create(
                            thread_id=thread.id,
                            role="user",
                            content=server_query
                        )
                        
                        # Run and get response
                        run = await ai_client.agents.runs.create(
                            thread_id=thread.id,
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
                                thread_id=thread.id,
                                run_id=run.id
                            )
                        
                        if run.status == "completed":
                            # Get messages
                            messages = ai_client.agents.messages.list(thread_id=thread.id)
                            
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
                        
                        # # Clean up thread
                        # try:
                        #     await ai_client.agents.threads.delete(thread_id=thread.id)
                        # except:
                        #     pass
                
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
                # Get the answer agent
                if not agent_id:
                    return []
                
                # Query the agent
                from azure.ai.projects.aio import AIProjectClient
                from azure.identity.aio import DefaultAzureCredential
                
                endpoint = os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
                
                async with DefaultAzureCredential(exclude_shared_token_cache_credential=True) as creds:
                    async with AIProjectClient(credential=creds, endpoint=endpoint) as ai_client:
                        # Create a thread for this query
                        thread = await ai_client.agents.threads.create()
                        
                        # Send the query
                        await ai_client.agents.messages.create(
                            thread_id=thread.id,
                            role="user",
                            content=query
                        )
                        
                        # Run and get response
                        run = await ai_client.agents.runs.create(
                            thread_id=thread.id,
                            agent_id=agent_id
                        )
                        
                        # Wait for completion
                        import asyncio
                        while run.status in ["queued", "in_progress", "requires_action"]:
                            await asyncio.sleep(1)
                            run = await ai_client.agents.runs.get(
                                thread_id=thread.id,
                                run_id=run.id
                            )
                        
                        if run.status == "completed":
                            # Get messages
                            messages = ai_client.agents.messages.list(thread_id=thread.id)
                            
                            # Parse the assistant's response
                            async for message in messages:
                                if message.role == "assistant":
                                    content = message.content[0].text.value if message.content else ""
                                    
                                    # Try to parse as JSON array
                                    try:
                                        import json
                                        dependencies = json.loads(content)
                                        if isinstance(dependencies, list):
                                            return dependencies
                                        elif isinstance(dependencies, dict) and "Response" in dependencies:
                                            # Handle case where agent returns in Response/Confidence/Citation format
                                            response_text = dependencies.get("Response", "")
                                            try:
                                                return json.loads(response_text)
                                            except:
                                                # Try to extract structured data from response
                                                return self._parse_dependency_text(response_text, server_name)
                                    except:
                                        # Fallback parsing
                                        return self._parse_dependency_text(content, server_name)
                        
                        # # Clean up
                        # await ai_client.agents.threads.delete(thread_id=thread.id)
                        
            except Exception as ex:
                logger.error(f"Failed to query dependencies for {server_name}: {ex}")
                return []

        def _parse_dependency_text(self, text: str, server_name: str) -> List[Dict]:
            """Parse dependency information from text response."""
            dependencies = []
            
            import re
            
            # Try to parse table format with more specific patterns
            lines = text.split('\n')
            
            for line in lines:
                # Skip headers and empty lines
                if 'Source' in line and 'Destination' in line:
                    continue
                if not line.strip():
                    continue
                
                # Pattern for dependency table entries
                # Example: POS System 192.168.50.10 API-GW-01 10.10.2.10 443/TCP Purpose
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
            
            # If no matches found with the table pattern, try simpler patterns
            if not dependencies:
                # Pattern for "Source: X Destination: Y Port: Z" format
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

    async def _process_questions_for_table(client_obj, agent_id: str, table_name: str, partition_key: str) -> dict:
        """Process all questions for a specific table and calculate confidence scores."""
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
        
        # Get agent
        try:
            agent_def = await client_obj.agents.get_agent(agent_id)
            answer_agent = SKAgent(client=client_obj, definition=agent_def)
        except Exception as ex:
            return {"result": "error", "message": f"Wrap agent failed: {ex}"}

        answered = 0
        low_conf_count = 0
        low_confidence_questions = []
        
        for ent in entities:
            q = ent.get("Question")
            if not q:
                continue
            
            try:
                # Get response from agent
                resp = await answer_agent.get_response(messages=q, thread=None)
                text = str(resp).strip()
                logger.debug(f"Agent response for '{q[:50]}...': in table {table_name}: {text[:200]}")
                
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
        """Process questions in all QA tables (AppDetails and PrivacyAndSecurity)."""
        tables_to_process = [
            f"AppDetails{application_id}",
            f"PrivacyAndSecurity{application_id}"
        ]
        
        all_results = {}
        all_low_confidence = []
        
        for table_name in tables_to_process:
            print(f"  Processing table: {table_name}...")
            result = await _process_questions_for_table(client_obj, agent_id, table_name, application_id)
            all_results[table_name] = result
            
            if result.get("result") == "ok":
                print(f"    ✓ Processed {result.get('answered')} questions")
                if result.get('lowConfidence', 0) > 0:
                    print(f"    ⚠ Low confidence: {result.get('lowConfidence', 0)}")
                
                # Collect all low confidence questions
                all_low_confidence.extend(result.get("lowConfidenceQuestions", []))
        
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
            
            # Startup sequence
            print(f"\n=== Starting Application Setup for '{application_id}' ===\n")
            
            # Step 1: Clone all template tables
            print("Step 1: Creating application-specific tables...")
            plugin = OrchestratorPlugin()
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
                    if agent_id:
                        print(f"✓ Answer agent created: {agent_id}")
                        
                        print("\nProcessing questions from all QA tables...")
                        qa_result = await _process_all_qa_tables(client, agent_id)
                        
                        if qa_result.get("result") == "ok":
                            for table_name, table_result in qa_result.get("results", {}).items():
                                print(f"  {table_name}:")
                                print(f"    - Processed: {table_result.get('answered')} questions")
                                print(f"    - Low confidence: {table_result.get('lowConfidence', 0)}")
                            
                            # Collect all low confidence questions
                            all_low_conf = qa_result.get("all_low_confidence_questions", [])
                            
                            # Step 6: Process dependency information
                            print("\nStep 6: Extracting and populating dependency information...")
                            dep_result = await plugin.populate_dependency_table()
                            dep_data = json.loads(dep_result)
                            if dep_data.get("result") == "ok":
                                print(f"✓ Populated {dep_data.get('populated')} dependency records")
                                print(f"  Processed {dep_data.get('servers_processed')} unique servers")
                            else:
                                print(f"⚠ Dependency extraction: {dep_data.get('message')}")
                        
                        # Step 7: Resolve low confidence questions if any exist
                        if len(all_low_conf) > 0:
                            print(f"\n✓ Total low confidence questions across all tables: {len(all_low_conf)}")
                            resolve = input("\nWould you like to resolve low confidence questions now? (yes/no): ").strip().lower()
                            if resolve in ["yes", "y"]:
                                await _interactive_low_confidence_resolution(all_low_conf)
            
            print(f"\n=== Setup Complete ===")
            print(f"Orchestrator ready. Type 'help' for available commands or 'exit' to quit.\n")
            
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
                        print("  /resolvelow - Resolve low confidence questions")
                        print("  /status - Show current status")
                        print("  /qasummary - Show Q&A summary")
                        print("  help - Show this help")
                        print("  exit - Quit the application\n")
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
                        
                        for table_name in tables:
                            try:
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
                    
                    # Normal chat
                    try:
                        response = await agent.get_response(messages=user_input, thread=thread)
                        print(f"Assistant: {response}")
                        thread = response.thread
                    except Exception as ex:
                        print(f"Assistant error: {ex}")
                        
            finally:
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
    args = parser.parse_args()
    
    import asyncio as _asyncio
    _asyncio.run(chat_loop(args.orchestrator_name, args.application_id))


if __name__ == "__main__":
    main()
