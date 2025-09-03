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
    
    logger.info(f"Triggering indexing function for appId={app_id}, container={container_name}")
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
                    logger.info(f"Created table: {target}")
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
                    logger.info(f"Indexing successful: {uploaded} documents uploaded")
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

    async def _process_questions_with_confidence(client_obj, agent_id: str) -> dict:
        """Process all questions and calculate confidence scores."""
        nonlocal last_answer_thread_id, low_confidence_questions
        
        try:
            from azure.data.tables import TableServiceClient, UpdateMode as TableUpdateMode
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            from semantic_kernel.agents import AzureAIAgent as SKAgent
        except Exception as ex:
            return {"result": "error", "message": f"Missing deps: {ex}"}

        qa_table = f"AppDetails{application_id}"
        qa_pk = application_id
        
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if conn_str:
            tsc = TableServiceClient.from_connection_string(conn_str)
        else:
            tables_url = os.getenv("AZURE_TABLES_ACCOUNT_URL")
            if not tables_url:
                return {"result": "unverified", "reason": "Missing AZURE_TABLES_ACCOUNT_URL"}
            cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
            tsc = TableServiceClient(endpoint=tables_url, credential=cred)

        tc = tsc.get_table_client(table_name=qa_table)
        
        # Get all entities
        escaped_pk = str(qa_pk).replace("'", "''")
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
        low_confidence_questions = []  # Reset the list
        
        for ent in entities:
            q = ent.get("Question")
            if not q:
                continue
            
            try:
                # Get response from agent
                resp = await answer_agent.get_response(messages=q, thread=None)
                text = str(resp).strip()
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
                
                # Calculate confidence based on response quality
                confidence = 0.0
                if parsed and isinstance(parsed, dict):
                    response_text = parsed.get("Response") or parsed.get("response") or ""
                    citation = parsed.get("Citation") or parsed.get("citation") or ""
                    
                    # Determine confidence level
                    if response_text and citation:
                        # Has both response and citation - high confidence
                        confidence = 0.8
                    elif response_text:
                        # Has response but no citation - medium confidence
                        confidence = 0.5
                    else:
                        # No response - low confidence
                        confidence = 0.2
                    
                    # Update entity
                    ent["Response"] = response_text
                    ent["Guidance"] = parsed.get("Guidance") or parsed.get("guidance") or ent.get("Guidance", "")
                    ent["Confidence"] = confidence
                    ent["Citation"] = citation
                else:
                    # Fallback if JSON parsing failed
                    ent["Response"] = text
                    ent["Confidence"] = 0.3
                    ent["Citation"] = ""
                
                # Track low confidence questions
                if ent["Confidence"] < 0.5:
                    low_confidence_questions.append({
                        "RowKey": ent.get("RowKey"),
                        "Question": q,
                        "Response": ent.get("Response", ""),
                        "Confidence": ent.get("Confidence", 0.0)
                    })
                    low_conf_count += 1
                
                # Update table
                tc.upsert_entity(entity=ent, mode=TableUpdateMode.MERGE)
                answered += 1
                
            except Exception as ex:
                logger.warning(f"Failed to process question for row {ent.get('RowKey')}: {ex}")
        
        return {
            "result": "ok",
            "answered": answered,
            "total": len(entities),
            "lowConfidence": low_conf_count,
            "lowConfidenceQuestions": low_confidence_questions
        }

    async def _interactive_low_confidence_resolution() -> None:
        """Interactively resolve low confidence answers with user."""
        nonlocal low_confidence_questions
        
        if not low_confidence_questions:
            print("No low confidence questions to resolve.")
            return
        
        print(f"\n=== Resolving {len(low_confidence_questions)} Low Confidence Questions ===")
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
            
            qa_table = f"AppDetails{application_id}"
            tc = tsc.get_table_client(table_name=qa_table)
            
            for idx, lc_q in enumerate(low_confidence_questions, 1):
                print(f"\nQuestion {idx}/{len(low_confidence_questions)}:")
                print(f"Q: {lc_q['Question']}")
                print(f"Current Answer (Confidence: {lc_q['Confidence']:.2f}): {lc_q['Response']}")
                
                try:
                    user_answer = input("\nProvide a better answer (or press Enter to keep current): ").strip()
                    
                    if user_answer:
                        # Update the table with user's answer
                        entity = tc.get_entity(partition_key=application_id, row_key=lc_q['RowKey'])
                        entity["Response"] = user_answer
                        entity["Confidence"] = 0.9  # High confidence for user-provided answers
                        entity["Citation"] = "User provided"
                        tc.upsert_entity(entity=entity, mode=TableUpdateMode.MERGE)
                        print("✓ Answer updated successfully")
                    else:
                        print("✓ Keeping current answer")
                        
                except KeyboardInterrupt:
                    print("\n\nSkipping remaining questions...")
                    break
                except Exception as ex:
                    print(f"Failed to update answer: {ex}")
            
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
            
            # Step 1: Clone template table
            print("Step 1: Creating application-specific Q&A table...")
            plugin = OrchestratorPlugin()
            clone_result = plugin.clone_template_table()
            clone_data = json.loads(clone_result)
            if clone_data.get("result") == "ok":
                print(f"✓ Table created: {clone_data.get('message')}")
            elif clone_data.get("result") == "exists":
                print(f"✓ Table already exists: {clone_data.get('table')}")
            else:
                print(f"✗ Failed to create table: {clone_data}")
            
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
                        
                        print("\nProcessing all questions...")
                        result = await _process_questions_with_confidence(client, agent_id)
                        print(f"✓ Processed {result.get('answered')} questions")
                        print(f"  Low confidence questions: {result.get('lowConfidence', 0)}")
                        
                        # Step 6: Resolve low confidence questions
                        if result.get('lowConfidence', 0) > 0:
                            resolve = input("\nWould you like to resolve low confidence questions now? (yes/no): ").strip().lower()
                            if resolve in ["yes", "y"]:
                                await _interactive_low_confidence_resolution()
            
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
                            result = await _process_questions_with_confidence(client, agent_id)
                            print(f"QA processing: {result}")
                        else:
                            print("Failed to create/get agent")
                        continue
                    
                    # Resolve low confidence
                    if user_input.startswith("/resolvelow"):
                        await _interactive_low_confidence_resolution()
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
