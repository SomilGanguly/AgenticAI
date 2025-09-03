import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import OpenApiTool, AzureAISearchTool, OpenApiAuthDetails
from azure.ai.projects.models import ConnectionType
import yaml

def create_code_analyzer_agent():
    load_dotenv()

    PROJECT_ENDPOINT = os.getenv("AGENT_ENDPOINT")
    MODEL_DEPLOYMENT = os.getenv("OPENAI_MODEL", "gpt-4o")
    FUNCTION_APP_URL = os.getenv("TERRAFORM_FUNCTION_APP_URL")
    FUNCTION_KEY = os.getenv("TERRAFORM_FUNCTION_KEY")
    SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=False)
    )

    # Load OpenAPI spec (as dict)
    with open("openapi.yaml", "r") as f:
        openapi_spec = yaml.safe_load(f)

    FUNCTION_KEY = os.getenv("TERRAFORM_FUNCTION_KEY")

    # Replace placeholder in parameter defaults
    if FUNCTION_KEY:
        for path_name, path_item in openapi_spec.get("paths", {}).items():
            if isinstance(path_item, dict):
                for method, op in path_item.items():
                    if isinstance(op, dict):
                        params = op.get("parameters", [])
                        for p in params:
                            if p.get("name") == "code":
                                schema = p.get("schema", {})
                                if schema.get("default") == "REPLACE_WITH_FUNCTION_KEY":
                                    schema["default"] = FUNCTION_KEY

    # Ensure server URL ends with /api without query
    if "servers" not in openapi_spec or not openapi_spec["servers"]:
        openapi_spec["servers"] = [{"url": FUNCTION_APP_URL.rstrip('/') + '/api'}]
    else:
        base_url = (FUNCTION_APP_URL or openapi_spec["servers"][0].get("url", "")).split('?')[0].rstrip('/')
        if not base_url.endswith("/api"):
            base_url += "/api"
        openapi_spec["servers"][0]["url"] = base_url

    print("DEBUG server url:", openapi_spec["servers"][0]["url"])
    print("DEBUG openapi paths:", list(openapi_spec.get("paths", {}).keys()))

    # NO manual ?code= in server, rely on parameter default
    terraform_tool = OpenApiTool(
        name="terraformAnalysisTools",
        description="Functions: github-fetcher and terraform-analyze for Terraform security assessment",
        spec=openapi_spec,
        auth=OpenApiAuthDetails(type="anonymous")
    )

    instructions_with_search = """You are a Terraform security analysis agent.
Follow this sequence strictly:
1. Extract exactly one GitHub repo URL from the user input (ask if missing).
2. Call github-fetcher { "github_url": "<url>" }.
3. Choose target_directory:
   - Prefer real deployment/env directories over module/example/test/sample/.terraform paths.
   - If multiple, pick highest tf_count unless user specifies.
4. If a *.tfvars file exists in that directory, read its raw content (from fetcher results) for tfvars_content.
5. Only include variable_overrides if explicitly provided by user.
6. Call terraform-analyze with repo_id, target_directory (+ tfvars_content / variable_overrides if applicable).
7. Build resource inventory from terraform_json.planned_values.root_module.resources (and any child_modules recursively).
8. Derive security issues (encryption, network exposure, identity, logging, replication, public access, missing policies).
9. For each issue, query the Search index with concise provider/resource/attribute baseline queries.
10. Output JSON:
{
  "security_gaps":[
    {
      "resource_type":"...",
      "resource_name":"...",
      "security_issue":"...",
      "severity":"high|medium|low",
      "baseline_reference":"...",
      "recommendation":"...",
      "terraform_fix":"..."
    }
  ],
  "summary":{
    "total_issues":N,
    "high_severity":N,
    "medium_severity":N,
    "low_severity":N
  }
}
Rules:
- Do NOT invent resources or attributes.
- If no issues, return empty security_gaps and zeroed summary.
- If a tool call fails, report failure and stop.
- Use Search only after terraform_json is obtained.
"""

    instructions_no_search = """You are a Terraform security analysis agent (no baseline search available).
Process:
1. Get repo URL -> github-fetcher.
2. Pick best deployment directory (exclude examples/tests).
3. Include tfvars_content if present.
4. Call terraform-analyze.
5. Generate issues only from actual resource attributes.
6. Output same JSON schema; set baseline_reference to "N/A".
If no issues, return empty list and zeroed summary counts.
"""

    try:
        # Azure AI Search connection (optional)
        azure_ai_conn_id = project_client.connections.get_default(ConnectionType.AZURE_AI_SEARCH).id
        search_tool = AzureAISearchTool(
            index_connection_id=azure_ai_conn_id,
            index_name=SEARCH_INDEX_NAME,
            query_type="simple",
            top_k=20
        )
        all_tools = [*terraform_tool.definitions, *search_tool.definitions]
        tool_resources = search_tool.resources if search_tool.resources else None
        instructions = instructions_with_search
    except Exception:
        all_tools = [*terraform_tool.definitions]
        tool_resources = None
        instructions = instructions_no_search

    agent = project_client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="code_analyzer_agent",
        instructions=instructions,
        tools=all_tools,
        tool_resources=tool_resources
    )

    print(f"Created code analyzer agent: {agent.name} (ID: {agent.id})")
    print("\nAdd to .env:")
    print(f"CODE_ANALYZER_AGENT_ID={agent.id}")
    return agent.id

if __name__ == "__main__":
    create_code_analyzer_agent()