from typing import Optional

import argparse
import asyncio
import logging
import os
import sys

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.ai.agents.models import AsyncToolSet, AzureAISearchTool, AzureAISearchQueryType
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv

from logging_config import configure_logging


load_dotenv()
logger = configure_logging(os.getenv("APP_LOG_FILE", ""))


async def ensure_agent(agent_name_input: Optional[str]) -> str:
    """Find an agent by name or create a new one with that name. Returns the agent ID."""
    endpoint = os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
    if not endpoint:
        raise RuntimeError("AZURE_EXISTING_AIPROJECT_ENDPOINT is not set")

    # Resolve the desired agent name: CLI input takes priority, then env var
    agent_name = agent_name_input or os.environ.get("AZURE_AI_AGENT_NAME")
    model_deployment = os.environ.get("AZURE_AI_AGENT_DEPLOYMENT_NAME")
    if not model_deployment:
        raise RuntimeError("AZURE_AI_AGENT_DEPLOYMENT_NAME must be set to create or use an agent")
    if not agent_name:
        raise RuntimeError("Agent name must be provided via --agent-name (or --application-id) or AZURE_AI_AGENT_NAME")

    async with DefaultAzureCredential(exclude_shared_token_cache_credential=True) as creds:
        async with AIProjectClient(credential=creds, endpoint=endpoint) as ai_client:
            # Try to find an agent by name if a name is provided
            existing_agent = None
            if agent_name:
                agent_list = ai_client.agents.list_agents()
                if agent_list:
                    async for agent_obj in agent_list:
                        if agent_obj.name == agent_name:
                            existing_agent = agent_obj
                            logger.info(f"Found existing agent named '{agent_obj.name}', ID: {agent_obj.id}")
                            os.environ["AZURE_EXISTING_AGENT_ID"] = agent_obj.id
                            break

            # Configure tool (optional) and instructions; create or update agent accordingly
            search_tool = None
            try:
                default_conn = await ai_client.connections.get_default(ConnectionType.AZURE_AI_SEARCH)
                if default_conn and getattr(default_conn, "id", None):
                    # Prefer explicit env; otherwise derive from agent name
                    index_name = os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", agent_name)

                    # Optional: verify the index exists if endpoint is provided (uses API key if set, else AAD)
                    def _check_index_exists(idx: str) -> Optional[bool]:
                        try:
                            svc_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
                            api_key = os.getenv("AZURE_SEARCH_API_KEY")
                            if not svc_endpoint:
                                return None
                            from azure.search.documents.indexes import SearchIndexClient
                            # Prefer API key if supplied; otherwise fall back to AAD via DefaultAzureCredential (sync)
                            if api_key:
                                from azure.core.credentials import AzureKeyCredential
                                credential = AzureKeyCredential(api_key)
                            else:
                                from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                                credential = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                            sic = SearchIndexClient(endpoint=svc_endpoint, credential=credential)
                            # Will raise ResourceNotFoundError if missing
                            sic.get_index(idx)
                            return True
                        except Exception as ex:
                            # Distinguish not found from other errors when possible
                            if ex.__class__.__name__ == "ResourceNotFoundError":
                                return False
                            logger.warning("Search index existence check failed: %s", ex)
                            return None

                    exists = _check_index_exists(index_name)
                    if exists is True:
                        # Choose query type based on available features
                        query_type = AzureAISearchQueryType.SIMPLE
                        try:
                            # Use SEMANTIC_HYBRID only if semantic types likely exist
                            from azure.search.documents.indexes.models import SemanticSettings as _SS  # type: ignore
                            query_type = AzureAISearchQueryType.SEMANTIC_HYBRID
                        except Exception:
                            pass

                        # Dynamic filter: default to appId == agent_name; allow override via AZURE_AI_SEARCH_FILTER
                        dynamic_filter = os.environ.get("AZURE_AI_SEARCH_FILTER") or f"appId eq '{agent_name}'"
                        search_tool = AzureAISearchTool(
                            index_connection_id=default_conn.id,
                            index_name=index_name,
                            query_type=query_type,
                            #filter=dynamic_filter,
                            filter="appId eq 'app01'",
                            top_k=10
                        )
                        logger.info(
                            "Configured Azure AI Search tool with index '%s' (query_type=%s)",
                            index_name,
                            getattr(query_type, "name", str(query_type)),
                        )
                    else:
                        logger.warning(
                            "Azure AI Search index '%s' not found or could not be verified; skipping tool attachment.",
                            index_name,
                        )
            except Exception as e:
                logger.warning("Azure AI Search tool not configured: %s", e)

            # Set instructions to ensure consistent JSON output for Q&A processing
            base_json_contract = (
                "For every question, return ONLY a strict JSON object with the exact keys: "
                "Response (string), Guidance (string), Confidence (number 0..1), Citation (string). "
                "Put where you found the answer in Citation. Do not include any text outside the JSON."
            )
            if search_tool is not None:
                # Enforce retrieval-only + strict JSON contract, and specify how to fill Confidence/Citation from search results
                instructions_text = (
                    """You MUST NOT invent answers
1. You MUST retrieve answers ONLY from the Azure AI Search tool. Do not use general knowledge.
2. Always call the search tool for EVERY question and use the highest-quality result(s).
3. If no relevant result is found, return empty strings for Response and Citation and 0 for Confidence.
4. Confidence MUST reflect the Azure AI Search retrieval score normalized to 0..1 (use @search.score or similar if available; otherwise use a conservative estimate like 0.4 for weak evidence).
5. Citation MUST be the document reference from Azure AI Search results (e.g., metadata_storage_name, id, source, or a clear document/file name). If multiple documents are used, include them separated by '; '.
6. Return all these in a json format.
"""
                )
            else:
                # No search tool available; still enforce non-hallucination and citation requirements
                instructions_text = (
                    f"You are the answer agent for application '{agent_name}'. "
                    "You MUST NOT invent answers. Only answer if you can cite an authoritative source you have access to; otherwise leave Response and Citation empty and set Confidence to 0. "
                    + base_json_contract
                )

            if existing_agent is not None:
                # Update existing agent with new instructions/toolset so it uses retrieval
                try:
                    update_kwargs = dict(
                        agent_id=existing_agent.id,
                        instructions=instructions_text,
                    )
                    if search_tool is not None:
                        toolset = AsyncToolSet()
                        toolset.add(search_tool)
                        update_kwargs["toolset"] = toolset
                    await ai_client.agents.update_agent(**update_kwargs)
                    logger.info("Updated existing agent instructions/toolset")
                except Exception as ex:
                    logger.warning(f"Update agent failed, will reuse as-is: {ex}")
                return existing_agent.id
            else:
                logger.info("Creating new agent")
                create_kwargs = dict(
                    model=model_deployment,
                    name=agent_name,
                    instructions=instructions_text,
                )
                if search_tool is not None:
                    toolset = AsyncToolSet()
                    toolset.add(search_tool)
                    create_kwargs["toolset"] = toolset

                agent = await ai_client.agents.create_agent(**create_kwargs)
                logger.info(f"Created agent, ID: {agent.id}")
                os.environ["AZURE_EXISTING_AGENT_ID"] = agent.id
                return agent.id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or reuse an Azure AI Agent by name.")
    parser.add_argument("--application-id", "-a", "--agent-name", dest="agent_name", help="Agent name to use. If exists, it will be reused; otherwise a new agent will be created.")
    args = parser.parse_args()

    agent_name = args.agent_name
    if not agent_name and sys.stdin.isatty():
        try:
            agent_name = input("Enter Agent Name (required): ").strip() or None
        except Exception:
            agent_name = None
    if not agent_name:
        print("Agent name is required. Provide --agent-name (or --application-id) or set AZURE_AI_AGENT_NAME.", flush=True)
        sys.exit(1)

    print("Initializing agent...", flush=True)
    agent_id = asyncio.run(ensure_agent(agent_name))
    print(f"Agent ID: {agent_id}", flush=True)


if __name__ == "__main__":
    main()