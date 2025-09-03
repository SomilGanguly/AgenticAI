import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import ConnectedAgentTool

# Tracing
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.azure_sdk import AzureSDKInstrumentor

def create_orchestrator_agent():
    load_dotenv()
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if conn:
        configure_azure_monitor(connection_string=conn)
        RequestsInstrumentor().instrument()
        AzureSDKInstrumentor().instrument()
    tracer = trace.get_tracer("orchestrator.setup")

    with tracer.start_as_current_span("initialize"):
        PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
        MODEL_DEPLOYMENT = os.getenv("OPENAI_MODEL", "gpt-4o")
        ARCHITECTURE_AGENT_ID = os.getenv("ARCHITECTURE_AGENT_ID")  # Set this in your .env
        CODE_ANALYZER_AGENT_ID = os.getenv("CODE_ANALYZER_AGENT_ID")  # Set this when ready

        project_client = AIProjectClient(
            endpoint=PROJECT_ENDPOINT,
            credential=DefaultAzureCredential(exclude_interactive_browser_credential=False)
        )

        # Create connected agent tool for architecture-agent
        architecture_agent_tool = ConnectedAgentTool(
            id=ARCHITECTURE_AGENT_ID,
            name="architecture_agent_openapi",  
            description="Analyzes Azure DevOps (ADO) wiki URLs for architecture patterns, security risks, and compliance issues against cloud security control framework standards."
        )

        connected_agents = [architecture_agent_tool]

        # Always attempt to include code analyzer if ID provided
        if CODE_ANALYZER_AGENT_ID:
            code_analyzer_tool = ConnectedAgentTool(
                id=CODE_ANALYZER_AGENT_ID,
                name="code_analyzer_agent",
                description="Performs Terraform-based security gap analysis using fetch + analyze tools and baseline knowledge."
            )
            connected_agents.append(code_analyzer_tool)

        # Aggregate tool definitions
        all_connected_tools = []
        for agent in connected_agents:
            all_connected_tools.extend(agent.definitions)

        with tracer.start_as_current_span("create_orchestrator") as span:
            span.set_attribute("tool.count", len(all_connected_tools))
            orchestrator_agent = project_client.agents.create_agent(
                model=MODEL_DEPLOYMENT,
                name="SBD-orchestrator",
                instructions=(
                    "You are SBD-orchestrator. You route user requests to specialized connected agents.\n"
                    "Routing rules:\n"
                    "- GitHub repository URL (contains 'github.com'): invoke code_analyzer_agent exactly once with that URL.\n"
                    "- Azure DevOps wiki URL (contains 'dev.azure.com' and '_wiki'): invoke architecture_agent_openapi.\n"
                    "- If both types appear, call both agents (in any order) and wait for all results.\n"
                    "Response composition:\n"
                    "1. Do NOT analyze content yourself—only aggregate the connected agents' outputs.\n"
                    "2. Preserve each agent's JSON structures; wrap them in a combined object:\n"
                    "{ \"architecture_analysis\": <agent_output?>, \"code_analysis\": <agent_output?> }\n"
                    "3. If an agent tool call fails, include an 'error' field for that section.\n"
                    "4. If URL type is unsupported, request clarification.\n"
                    "5. Never duplicate or modify the code analyzer's security_gaps or counts.\n"
                    "6. Do not re-query tools redundantly; one pass per URL type unless user asks for re-run."
                ),
                tools=all_connected_tools,
            )
            span.set_attribute("orchestrator.id", orchestrator_agent.id)

    print(f"Created orchestrator agent: {orchestrator_agent.name} (ID: {orchestrator_agent.id})")
    print(f"\nAdd this to your .env file:")
    print(f"ORCHESTRATOR_AGENT_ID={orchestrator_agent.id}")
    return orchestrator_agent.id

if __name__ == "__main__":
    orchestrator_id = create_orchestrator_agent()