import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import OpenApiTool, AzureAISearchTool, AzureAISearchQueryType, OpenApiAuthDetails
from azure.ai.projects.models import ConnectionType

def create_architecture_agent_with_openapi():
    load_dotenv()
    PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    MODEL_DEPLOYMENT = os.getenv("OPENAI_MODEL", "gpt-4o")
    INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")
    FUNCTION_URL = os.getenv("AZURE_FUNCTION_URL")  # Add this to .env
    FUNCTION_KEY = os.getenv("AZURE_FUNCTION_KEY")  # Add this to .env

    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=False)
    )

    # Azure AI Search setup
    azure_ai_conn_id = project_client.connections.get_default(ConnectionType.AZURE_AI_SEARCH).id
    ai_search = AzureAISearchTool(
        index_connection_id=azure_ai_conn_id,
        index_name=INDEX_NAME,
        query_type=AzureAISearchQueryType.SIMPLE,
        top_k=50,
        filter="",
    )

    # OpenAPI specification for your Azure Function
    # Updated to include the function key in the URL
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Wiki Data Extractor API",
            "version": "1.0.0"
        },
        "servers": [
            {
                "url": f"{FUNCTION_URL}",
                "variables": {
                    "functionKey": {
                        "default": FUNCTION_KEY,
                        "description": "Azure Function key for authentication"
                    }
                }
            }
        ],
        "paths": {
            "/api/wikiDataExtractor": {
                "post": {
                    "operationId": "extractWikiData",
                    "summary": "Extract text and images from Azure DevOps wiki pages",
                    "parameters": [
                        {
                            "name": "code",
                            "in": "query",
                            "required": True,
                            "schema": {
                                "type": "string",
                                "default": FUNCTION_KEY
                            },
                            "description": "Function key for authentication"
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "project_url": {
                                            "type": "string",
                                            "description": "The Azure DevOps wiki URL to extract content from"
                                        }
                                    },
                                    "required": ["project_url"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful extraction",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Extracted text content from the wiki"
                                            },
                                            "image_descriptions": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string"
                                                },
                                                "description": "AI-generated descriptions of images found in the wiki"
                                            },
                                            "total_images": {
                                                "type": "integer",
                                                "description": "Total number of images processed"
                                            },
                                            "status": {
                                                "type": "string",
                                                "description": "Status of the extraction"
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            }
        }
    }

    # Create OpenAPI tool with anonymous auth
    openapi_tool = OpenApiTool(
        name="wikiDataExtractor",
        description="Extracts text and analyzes images from Azure DevOps wiki pages",
        spec=openapi_spec,
        auth=OpenApiAuthDetails(type="anonymous")
    )

    # Combine tools
    all_tools = [*openapi_tool.definitions, *ai_search.definitions]

    # Create agent with OpenAPI tool
    agent = project_client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="architecture_agent_openapi",
        instructions=(
            "You are an expert architecture security agent with access to a cloud security control framework (SCF) knowledge base. "
            "Follow these steps EXACTLY:\n"
            "1. Use the wikiDataExtractor tool to extract all text and images from the provided Azure DevOps wiki URL.\n"
            "IMPORTANT: The wikiDataExtractor is now an HTTP API - call it with the project_url parameter."
            "2. After extraction, use the Azure AI Search tool MULTIPLE times with different search queries:\n"
            "   - Search for 'SCF'\n"
            "3. The search index contains documents with a 'wiki_path' field that references the location of security standards.\n"
            "4. Compare the extracted design patterns against ALL relevant security standards found in the search results.\n"
            "5. For EVERY discrepancy, non-compliance, or risk found, create a JSON object with:\n"
            "   - deficiency_id: A unique identifier\n"
            "   - severity: High/Medium/Low\n"
            "   - status: 'Open'\n"
            "   - current_date: Today's date in YYYY-MM-DD format\n"
            "   - deficiency_type: Type of security issue\n"
            "   - reference: The exact 'wiki_path' from the search index where the violated standard is documented\n"
            "   - owner: Team/person responsible (if identifiable from wiki)\n"
            "   - affected_assets: Components affected\n"
            "   - deficiency_title: Brief description\n"
            "   - threat_description: Detailed security threat\n"
            "   - proposed_mitigation: Specific remediation steps\n"
            "6. For EVERY discrepancy, non-compliance, or risk found, create a JSON object with the required fields.\n"
            "7. Return ONLY the JSON array of issues. Use empty strings for unknown fields.\n"
            "IMPORTANT: The index contains SCF data - you MUST find and use it for comparison."

        ),
        tools=all_tools,
        tool_resources=ai_search.resources if ai_search.resources else {}
    )
    
    print(f"Created agent with OpenAPI tool: {agent.name} (ID: {agent.id})")
    print(f"\nAdd this to your .env file:")
    print(f"ARCHITECTURE_AGENT_OPENAPI_ID={agent.id}")
    
    return agent.id

if __name__ == "__main__":
    create_architecture_agent_with_openapi()
