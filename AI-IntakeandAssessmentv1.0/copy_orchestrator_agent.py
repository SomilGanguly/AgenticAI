from typing import Optional

import argparse
import json
import logging
import os
import sys

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

        @kernel_function(description="Read a field from Azure Table Storage to determine indexing status. If no entity exists for the Application ID, create it with IndexingStatus=false and ContainerCreated=true. Table and partition key come from env when not provided: AZURE_INDEX_STATUS_TABLE_NAME, AZURE_INDEX_STATUS_PARTITION_KEY. Uses AZURE_STORAGE_CONNECTION_STRING or DefaultAzureCredential with AZURE_TABLES_ACCOUNT_URL.")
        def get_indexing_status(self, table_name: Optional[str] = None, partition_key: Optional[str] = None, row_key: Optional[str] = None, field_name: Optional[str] = "IndexingStatus", account_url: Optional[str] = None) -> str:
            try:
                table = table_name or os.getenv("AZURE_INDEX_STATUS_TABLE_NAME") or "indexingStatus"
                pk = partition_key or os.getenv("AZURE_INDEX_STATUS_PARTITION_KEY", "OnPremToAzureMigration")
                rk = row_key or application_id
                logger.debug(f"Indexing status lookup: table={table}, pk={pk}, rk={rk}, field={field_name}")
                missing = [n for n, v in {"table_name": table, "partition_key": pk, "row_key": rk}.items() if not v]
                if missing:
                    return json.dumps({"result": "unverified", "reason": f"Missing values: {', '.join(missing)}"})

                conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                try:
                    from azure.data.tables import TableServiceClient
                except Exception as ex:
                    return json.dumps({"result": "error", "message": f"TableServiceClient import failed: {ex}"})

                if conn_str:
                    try:
                        tsc = TableServiceClient.from_connection_string(conn_str)
                    except Exception as ex:
                        return json.dumps({"result": "error", "message": f"TableServiceClient init failed: {ex}"})
                else:
                    try:
                        from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                        tables_url = account_url or os.getenv("AZURE_TABLES_ACCOUNT_URL")
                        if not tables_url:
                            return json.dumps({"result": "unverified", "reason": "Missing AZURE_TABLES_ACCOUNT_URL"})
                        cred = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                        tsc = TableServiceClient(endpoint=tables_url, credential=cred)
                    except Exception as ex:
                        return json.dumps({"result": "error", "message": f"TableServiceClient init failed: {ex}"})

                try:
                    tc = tsc.get_table_client(table_name=table)
                    entity = tc.get_entity(partition_key=pk, row_key=rk)
                except Exception as ex:
                    if ex.__class__.__name__ in {"ResourceNotFoundError", "ResourceNotFound"}:
                        # Create table if missing, then create the entity with default fields
                        try:
                            # Best-effort create-if-not-exists
                            try:
                                create_if = getattr(tsc, "create_table_if_not_exists", None)
                                if callable(create_if):
                                    create_if(table_name=table)
                                else:
                                    try:
                                        tsc.create_table(table_name=table)
                                    except Exception as ce:
                                        if "TableAlreadyExists" not in str(ce):
                                            return json.dumps({"result": "table_create_failed", "message": str(ce), "table": table})
                            except Exception as ce:
                                if "TableAlreadyExists" not in str(ce):
                                    return json.dumps({"result": "table_create_failed", "message": str(ce), "table": table})

                            # Upsert default row
                            logger.debug("Entity not found; upserting default row with IndexingStatus=false, ContainerCreated=true")
                            entity_to_create = {
                                "PartitionKey": pk,
                                "RowKey": rk,
                                "IndexingStatus": False,
                                "ContainerCreated": True,
                            }
                            tc = tsc.get_table_client(table_name=table)
                            tc.upsert_entity(entity=entity_to_create)
                            value = entity_to_create.get(field_name)
                            return json.dumps({
                                "result": "created",
                                "table": table,
                                "pk": pk,
                                "rk": rk,
                                "field": field_name,
                                "value": value,
                                "isComplete": bool(value) if isinstance(value, (bool, int, str)) else None,
                            })
                        except Exception as up_ex:
                            return json.dumps({"result": "create_failed", "message": str(up_ex), "table": table, "pk": pk, "rk": rk})
                    return json.dumps({"result": "unverified", "message": str(ex)})

                try:
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
                    return json.dumps({"result": "field_error", "message": str(ex)})
            except Exception as ex:
                return json.dumps({"result": "error", "message": str(ex)})
    async with DefaultAzureCredential(exclude_shared_token_cache_credential=True) as creds:
        async with AzureAIAgent.create_client(credential=creds, endpoint=endpoint) as client:
            # Create or reuse an orchestrator agent on the service
            # Load instructions from external file if available
            instructions_text = (
                f"You are an orchestrator. Application ID: '{application_id}'. Converse naturally. "
                "At the start of the session, ensure the required storage is ready and indexing status is known: "
                "1) Call the plugin function 'check_container_exists' (it will create the container if missing) and report the container URL to the user. "
                "2) Call the plugin function 'get_indexing_status' to read the status and summarize the result and next steps. "
                "When asked to create a target agent, ask the user to run /create <agent_name> <index_name> after the index exists."
            )
            try:
                instr_path = os.getenv("ORCHESTRATOR_INSTRUCTIONS_FILE", os.path.join(os.path.dirname(__file__), "orchestrator_instructions.md"))
                if os.path.isfile(instr_path):
                    with open(instr_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            instructions_text = f"Application ID: '{application_id}'.\n\n" + content
            except Exception:
                pass

            try:
                agent_definition = await client.agents.create_agent(
                    model=deployment,
                    name=orchestrator_name,
                    instructions=instructions_text,
                )
            except Exception as ex:
                print(f"Failed to create orchestrator agent: {ex}", flush=True)
                sys.exit(1)

            # Wrap with SK AzureAIAgent and attach plugin (for future auto-tooling)
            try:
                agent = AzureAIAgent(client=client, definition=agent_definition, plugins=[OrchestratorPlugin()])
            except Exception:
                agent = AzureAIAgent(client=client, definition=agent_definition)

            thread: Optional[AzureAIAgentThread] = None
            # Auto-run startup checks using plugin functions and show results
            try:
                startup_prompt = (
                    f"For application id '{application_id}', first ensure the blob container exists (create if missing) by calling orchestrator.check_container_exists(), "
                    f"then check indexing status by calling orchestrator.get_indexing_status(). Summarize the results and next steps for the user."
                )
                logger.debug("Sending startup prompt to agent for auto checks")
                startup_response = await agent.get_response(messages=startup_prompt, thread=thread)
                print(f"Assistant: {startup_response}")
                thread = startup_response.thread
            except Exception as ex:
                print(f"Startup checks failed: {ex}")

            print(f"Orchestrator (SK AzureAIAgent) ready. Application ID: '{application_id}'. Type 'exit' to quit.")
            try:
                while True:
                    try:
                        user_input = input("You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if user_input.lower() in {"exit", "quit"}:
                        break

                    # Fallback slash command: /create [agent_name] [index_name]
                    if user_input.startswith("/create"):
                        parts = user_input.split()
                        tgt_name = application_id if len(parts) < 2 else parts[1]
                        idx = os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", tgt_name) if len(parts) < 3 else parts[2]
                        status = check_index_exists(idx)
                        if status is True:
                            try:
                                new_agent_id = await ensure_agent(tgt_name)
                                print(f"Created/Reused agent '{tgt_name}' -> {new_agent_id}")
                            except Exception as ex:
                                print(f"Agent creation failed: {ex}")
                        elif status is False:
                            print(f"Index '{idx}' missing; not creating.")
                        else:
                            print(f"Could not verify index '{idx}'.")
                        continue

                    # Normal chat turn
                    try:
                        logger.debug("Sending user message to agent")
                        response = await agent.get_response(messages=user_input, thread=thread)
                        print(f"Assistant: {response}")
                        thread = response.thread
                    except Exception as ex:
                        print(f"Assistant error: {ex}")
                        continue
            finally:
                # Optional cleanup (keep agent for reuse; delete thread if created)
                try:
                    await thread.delete() if thread else None
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Orchestrator Agent (Semantic Kernel AzureAIAgent)")
    parser.add_argument(
        "--orchestrator-name",
        default=os.environ.get("ORCHESTRATOR_AGENT_NAME", "orchestrator"),
        help="Orchestrator agent name (default: 'orchestrator')",
    )
    args = parser.parse_args()
    # Run the async chat loop
    import asyncio as _asyncio
    _asyncio.run(chat_loop(args.orchestrator_name))


if __name__ == "__main__":
    main()
