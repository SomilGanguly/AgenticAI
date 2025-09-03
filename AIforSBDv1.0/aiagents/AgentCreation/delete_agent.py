import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

def delete_agent(agent_id):
    load_dotenv()
    PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=False)
    )
    project_client.agents.delete_agent(agent_id)
    print(f"Deleted agent with ID: {agent_id}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        delete_agent(sys.argv[1])
    else:
        print("Usage: python delete_agent.py <agent_id>")
