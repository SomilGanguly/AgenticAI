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


def _sanitize_index_name(raw: str) -> str:
    """Sanitize arbitrary application/agent name into a valid Azure AI Search index name.

    Rules enforced:
    - Lowercase only
    - Allowed chars: a-z, 0-9, -
    - Collapse multiple dashes
    - Trim leading/trailing dashes
    - Must start with alphanumeric (fallback to hash-based prefix if not)
    - Length 2..128 (truncate or pad as needed)
    """
    import re, hashlib
    s = (raw or "").lower().strip()
    s = re.sub(r"[^a-z0-9-]", "-", s)
    s = re.sub(r"-+", "-", s).strip('-')
    if not s or not s[0].isalnum():
        s = f"app-{hashlib.sha1((raw or 'x').encode()).hexdigest()[:8]}"
    if len(s) < 2:
        s = (s + "ix")[:2]
    if len(s) > 128:
        s = s[:128]
    return s


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
                            logger.debug(f"Found existing agent named '{agent_obj.name}', ID: {agent_obj.id}")
                            os.environ["AZURE_EXISTING_AGENT_ID"] = agent_obj.id
                            break

            # Configure tool (optional) and instructions; create or update agent accordingly
            search_tool = None
            try:
                # Get default AI Search connection
                default_conn = await ai_client.connections.get_default(ConnectionType.AZURE_AI_SEARCH)
                if default_conn and getattr(default_conn, "id", None):
                    # Sanitize agent name for use as search index name
                    index_name = _sanitize_index_name(agent_name)
                    if index_name != agent_name:
                        logger.debug(f"Sanitized index name '{index_name}' from agent name '{agent_name}'")

                    # Optional: verify the index exists if endpoint is provided (uses API key if set, else AAD)
                    def _check_index_exists(idx: str) -> Optional[bool]:
                        try:
                            # Use same env var names as indexer.py for consistency
                            svc_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") or os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
                            api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
                            if not svc_endpoint:
                                logger.debug("No search endpoint configured for index validation")
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

                    # Check if semantic configuration is available
                    def _check_semantic_config(idx: str) -> Optional[str]:
                        """Check if semantic search is configured and return config name."""
                        try:
                            # Use same env var names as indexer.py for consistency
                            svc_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") or os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
                            api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
                            sem_config = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")
                            
                            if not svc_endpoint:
                                logger.debug("No search endpoint configured for semantic config check")
                                return sem_config  # Return env var if we can't verify
                            
                            from azure.search.documents.indexes import SearchIndexClient
                            if api_key:
                                from azure.core.credentials import AzureKeyCredential
                                credential = AzureKeyCredential(api_key)
                            else:
                                from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                                credential = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                            
                            sic = SearchIndexClient(endpoint=svc_endpoint, credential=credential)
                            index = sic.get_index(idx)
                            
                            # Check if semantic search is configured
                            if hasattr(index, 'semantic_search') and index.semantic_search:
                                configs = getattr(index.semantic_search, 'configurations', [])
                                if configs and len(configs) > 0:
                                    # Return the first semantic config name or the env var
                                    config_name = getattr(configs[0], 'name', sem_config) or sem_config
                                    logger.debug(f"Found semantic configuration: {config_name}")
                                    return config_name
                            
                            logger.debug("No semantic search configuration found in index")
                            return sem_config
                        except Exception as ex:
                            logger.debug(f"Semantic config check failed: {ex}")
                            return os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")

                    exists = _check_index_exists(index_name)
                    if exists is True:
                        # Enhanced query type selection for semantic hybrid search
                        semantic_config = _check_semantic_config(index_name)
                        
                        # Check if the index has vector fields (indicates vector search capability)
                        def _check_vector_fields(idx: str) -> bool:
                            try:
                                # Use same env var names as indexer.py for consistency
                                svc_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") or os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
                                api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
                                if not svc_endpoint:
                                    logger.debug("No search endpoint configured for vector field check")
                                    return False
                                
                                from azure.search.documents.indexes import SearchIndexClient
                                if api_key:
                                    from azure.core.credentials import AzureKeyCredential
                                    credential = AzureKeyCredential(api_key)
                                else:
                                    from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
                                    credential = SyncDefaultAzureCredential(exclude_shared_token_cache_credential=True)
                                
                                sic = SearchIndexClient(endpoint=svc_endpoint, credential=credential)
                                index = sic.get_index(idx)
                                
                                # Check for vector fields - look for contentVector field that matches indexer.py
                                if hasattr(index, 'fields'):
                                    for field in index.fields:
                                        # Check for contentVector field created by indexer.py
                                        if (hasattr(field, 'name') and field.name == 'contentVector' and
                                            hasattr(field, 'type') and 
                                            'Collection(Edm.Single)' in str(field.type) and
                                            hasattr(field, 'vector_search_dimensions') and 
                                            field.vector_search_dimensions > 0):
                                            logger.debug(f"Found vector field '{field.name}' with {field.vector_search_dimensions} dimensions")
                                            return True
                                logger.debug("No vector fields found in index")
                                return False
                            except Exception as ex:
                                logger.debug(f"Vector field check failed: {ex}")
                                return False
                        
                        has_vector_fields = _check_vector_fields(index_name)
                        
                        # Select optimal query type based on available capabilities
                        if has_vector_fields and semantic_config:
                            # Best case: Use VECTOR_SEMANTIC_HYBRID for optimal results
                            query_type = AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID
                            logger.debug(f"Using VECTOR_SEMANTIC_HYBRID search (best quality) with config '{semantic_config}'")
                        elif semantic_config:
                            # Use SEMANTIC for semantic reranking
                            query_type = AzureAISearchQueryType.SEMANTIC
                            logger.debug(f"Using SEMANTIC search with config '{semantic_config}'")
                        elif has_vector_fields:
                            # Use VECTOR for similarity search
                            query_type = AzureAISearchQueryType.VECTOR
                            logger.debug("Using VECTOR search for similarity matching")
                        else:
                            # Fall back to SIMPLE keyword search
                            query_type = AzureAISearchQueryType.SIMPLE
                            logger.debug("Using SIMPLE keyword search")

                        # Dynamic filter: default to appId == agent_name
                        dynamic_filter = os.environ.get("AZURE_AI_SEARCH_FILTER") or f"appId eq '{agent_name}'"
                        logger.debug(f"Using search filter: {dynamic_filter}")

                        try:
                            # Create/update project index following Microsoft's pattern
                            # This registers the existing index with the AI project
                            project_index_name = f"project-index-{index_name}"
                            index_version = "1"
                            
                            # Define field mapping for semantic hybrid search
                            field_mapping = {
                                "contentFields": ["content", "metadata"],  # Multiple content fields for better semantic search
                                "titleField": "title",
                                "urlField": "path",  # Use path as URL field
                                "vectorFields": ["contentVector"] if has_vector_fields else []
                            }
                            
                            logger.debug(f"Creating/updating project index '{project_index_name}' with field mapping")
                            
                            # Get connection name - try different possible properties
                            connection_name = "default"
                            if hasattr(default_conn, 'name'):
                                connection_name = default_conn.name
                            elif hasattr(default_conn, 'connection_name'):
                                connection_name = default_conn.connection_name
                            elif hasattr(default_conn, 'id'):
                                # Use ID as connection name if name is not available
                                connection_name = default_conn.id
                            
                            logger.debug(f"Using connection name: {connection_name}")
                            
                            project_index = await ai_client.indexes.create_or_update(
                                name=project_index_name,
                                version=index_version,
                                index={
                                    "connectionName": connection_name,
                                    "indexName": index_name,  # Reference your existing index
                                    "type": "AzureSearch",
                                    "fieldMapping": field_mapping
                                }
                            )
                            
                            # Create search tool using the project index asset
                            # For project index approach: use index_asset_id only, with empty connection details
                            search_tool = AzureAISearchTool(
                                index_connection_id="",  # Empty for project index approach
                                index_name="",  # Empty for project index approach
                                query_type=query_type,
                                filter=dynamic_filter,
                                top_k=20,  # Return top 5 results for better quality
                                index_asset_id=f"{project_index.name}/versions/{project_index.version}"  # Project index reference
                            )
                            
                            logger.debug(
                                "Configured Azure AI Search tool with project index '%s' (query_type=%s, semantic=%s, vector=%s)",
                                project_index_name,
                                getattr(query_type, "name", str(query_type)),
                                "enabled" if semantic_config else "disabled",
                                "enabled" if has_vector_fields else "disabled"
                            )
                        except Exception as tool_ex:
                            logger.warning(f"Failed to create project index or search tool: {tool_ex}")
                            # Fallback to legacy connection-based approach
                            try:
                                logger.debug("Falling back to legacy search tool configuration")
                                search_tool = AzureAISearchTool(
                                    index_connection_id=default_conn.id,
                                    index_name=index_name,
                                    query_type=query_type,  # Use the optimally selected query type
                                    filter=dynamic_filter,
                                    top_k=5
                                )
                                logger.debug("Configured fallback search tool with connection ID")
                            except Exception as fallback_ex:
                                logger.error(f"Fallback search tool creation also failed: {fallback_ex}")
                                search_tool = None
                    else:
                        logger.warning(
                            "Azure AI Search index '%s' not found or could not be verified; skipping tool attachment.",
                            index_name,
                        )
            except Exception as e:
                logger.warning("Azure AI Search tool not configured: %s", e)

            # Enhanced instructions for hybrid/semantic search
            base_json_contract = (
                "For every question, return ONLY a strict JSON object with the exact keys: "
                "Response (string), Confidence (number 0..1), Citation (string). "
                "Put where you found the answer in Citation. Do not include any text outside the JSON."
            )
            
            if search_tool is not None:
                # Enhanced instructions optimized for semantic hybrid search
                instructions_text = (
                    f"""only use the attached tool to give response. dont generate your own data. You are the answer agent for application id '{agent_name}'(this is not application name).
ROLE: Pure retrieval + exact-value extraction with conflict detection. Never invent, infer beyond what is explicitly in search results.

OUTPUT (ALWAYS A SINGLE JSON OBJECT):
{{
  "Response": "<string or 'CONFLICT: [value1] vs [value2]...'>",
  "Confidence": <0.0..1.0>,
  "Citation": "<source document(s)>",
  "ConflictDetected": <true|false>,
  "ConflictingValues": [<list of different values>], // only if conflicts exist
  "SearchStrategy": "<semantic|vector|hybrid|keyword>" // indicate which search approach worked best
}}

CONFLICT DETECTION RULES:
1. When searching for specific values (IP addresses, ports, server names, configurations):
   - Collect ALL distinct values found across documents
   - If multiple DIFFERENT values exist for the same entity, mark as conflict
   - List all conflicting values in ConflictingValues array
   - Set Confidence to 0.2 or lower for conflicts
   - Format Response as: "CONFLICT DETECTED: Found multiple values: [value1] in [source1], [value2] in [source2]..."

2. Examples of conflicts to detect:
   - Different IP addresses for same server
   - Different port numbers for same service
   - Different configurations for same component
   - Contradictory yes/no answers for same question

SEARCH STRATEGY:
1. Perform comprehensive search using multiple keywords
2. Use semantic search to find all relevant documents
3. Extract ALL values related to the question from ALL documents
4. Compare values across documents
5. If values differ, treat as conflict

RULES:
1. Always search with the tool - never skip
2. Use semantic search capabilities to find related content
3. Extract exact values as they appear (no normalization unless identical)
4. For conflicts: list ALL different values found with their sources
5. Never pick one value over another when conflict exists
6. Set low confidence (0.2 or less) for any conflicting answers

EXAMPLES:
Question: "What is the IP address for APP-02?"
If found: 10.10.3.45 in doc1 and 10.10.3.46 in doc2
Response: {{
  "Response": "CONFLICT DETECTED: Found multiple IP addresses for APP-02: 10.10.3.45 (Application_Design_Document.docx), 10.10.3.46 (Server_Inventory.docx)",
  "Confidence": 0.2,
  "Citation": "Multiple sources",
  "ConflictDetected": true,
  "ConflictingValues": ["10.10.3.45", "10.10.3.46"]
}}

BEGIN. Respond only with the mandated JSON object for each user question."""
                )
            else:
                # No search tool available; still enforce non-hallucination and citation requirements
                instructions_text = (
                    f"You are the answer agent for application with application Id: '{agent_name}'. "
                    "You MUST NOT invent answers. Only answer if you can cite an authoritative source you have access to; otherwise leave Response and Citation empty and set Confidence to 0. "
                    + base_json_contract
                )

            if existing_agent is not None:
                # Update existing agent with new instructions/toolset so it uses retrieval
                try:
                    update_kwargs = dict(
                        agent_id=existing_agent.id,
                        instructions=instructions_text,
                        temperature=0.1  # Set low temperature for deterministic responses
                    )
                    if search_tool is not None:
                        # Use tools and tool_resources following Microsoft's pattern
                        update_kwargs["tools"] = search_tool.definitions
                        update_kwargs["tool_resources"] = search_tool.resources
                    await ai_client.agents.update_agent(**update_kwargs)
                    logger.debug("Updated existing agent instructions/tools for semantic hybrid search")
                except Exception as ex:
                    logger.warning(f"Update agent failed, will reuse as-is: {ex}")
                return existing_agent.id
            else:
                logger.debug("Creating new agent with semantic hybrid search capabilities")
                create_kwargs = dict(
                    model=model_deployment,
                    name=agent_name,
                    instructions=instructions_text,
                    temperature=0.1  # Set low temperature for deterministic responses
                )
                if search_tool is not None:
                    # Use tools and tool_resources following Microsoft's pattern
                    create_kwargs["tools"] = search_tool.definitions
                    create_kwargs["tool_resources"] = search_tool.resources

                agent = await ai_client.agents.create_agent(**create_kwargs)
                logger.debug(f"Created agent with semantic hybrid search, ID: {agent.id}")
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