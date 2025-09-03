# New Intake Agent

This script initializes or reuses an Azure AI Agent. You can provide an application/agent ID via CLI or interactively; if none is found, it will create a new agent.

## Prerequisites
- Python 3.10+
- Azure credentials available to `DefaultAzureCredential` (e.g., `az login` or managed identity)
- Environment variables:
  - `AZURE_EXISTING_AIPROJECT_ENDPOINT` (AI Project endpoint URL)
  - `AZURE_AI_AGENT_DEPLOYMENT_NAME` (model deployment name)
  - `AZURE_AI_AGENT_NAME` (required unless you pass --agent-name/--application-id to the script)
  - Optional: `AZURE_EXISTING_AGENT_ID` or `AZURE_AI_AGENT_ID` (existing agent ID)
  - Optional: `AZURE_AI_SEARCH_INDEX_NAME` (index name for Azure AI Search; defaults to `sample_index`)
  - Optional (for index pre-existence check):
    - `AZURE_SEARCH_SERVICE_ENDPOINT` (Search service endpoint URL)
    - `AZURE_SEARCH_API_KEY` (Admin API key; optional if using AAD via DefaultAzureCredential)
  - `AZURE_SEARCH_API_KEY` (optional if using AAD)

If your project has a default Azure AI Search connection, the script will automatically attach an Azure AI Search tool to the agent using that connection. By default, it uses `AZURE_AI_SEARCH_INDEX_NAME` if set; otherwise it derives the index name from the agent name you provide. The tool uses HYBRID query type when available. If `AZURE_SEARCH_SERVICE_ENDPOINT` and optionally `AZURE_SEARCH_API_KEY` are provided, the script will validate that the index exists; if it doesn't or can't be verified, the tool will not be attached and the agent will still be created.

## Install
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
- Prompt for agent name interactively:
```
python intake_agent.py
```
- Provide an agent name explicitly (reuses if exists, otherwise creates new):
```
python intake_agent.py --agent-name <YOUR_AGENT_NAME>
```

Logs write to stdout and optionally to a file if `APP_LOG_FILE` is set.

### Orchestrator (create agent only when index exists)
Use `orchestrator_agent.py` to ensure the Azure AI Search index is present before creating the agent:
```
python orchestrator_agent.py --agent-name <YOUR_AGENT_NAME> --index-name <INDEX_NAME>
```
- If `--index-name` is omitted, it defaults to the agent name or `AZURE_AI_SEARCH_INDEX_NAME`.
- The script verifies the index using `AZURE_SEARCH_SERVICE_ENDPOINT` and optionally `AZURE_SEARCH_API_KEY`.
- On success, it invokes `intake_agent.py` logic to create/reuse the agent; otherwise it skips creation.
- Add `--strict` to fail when the index cannot be verified instead of skipping creation.
