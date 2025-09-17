# Copyright (c) Microsoft. All rights reserved.
import os
import asyncio
from typing import Annotated
from typing import List
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType
from azure.identity.aio import AzureCliCredential
from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.functions import kernel_function
from dotenv import load_dotenv
load_dotenv()
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, FunctionResultContent


# Define a sample plugin for the sample
class AsrPromptPlugin:

    @kernel_function(description="Upload a file to an Azure Blob Storage container. The container name is the app id. Returns the blob URL.")
    def upload_file_to_container(self, file_path: Annotated[str, "Path to the file to upload."], app_id: Annotated[str, "App ID (container name)."], blob_name: Annotated[str, "Name for the blob in storage."] = None) -> Annotated[str, "Returns the blob URL."]:
        from azure.storage.blob import BlobServiceClient, ContentSettings
        import os
        account_url = os.getenv("AZURE_BLOB_ACCOUNT_URL") or os.getenv("AZURE_STORAGE_ACCOUNT_URL") or os.getenv("AZURE_TABLES_ACCOUNT_URL") or os.getenv("AZURE_TABLE_ACCOUNT_URL")
        if account_url:
            account_url = account_url.strip()
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        if not account_url and account_name:
            account_url = f"https://{account_name}.blob.core.windows.net"
        credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
        blob_service = BlobServiceClient(account_url, credential=credential)
        container_name = str(app_id).lower()
        if not blob_name:
            blob_name = os.path.basename(file_path)
        # Create container if not exists
        container_client = blob_service.get_container_client(container_name)
        try:
            container_client.create_container()
        except Exception:
            pass  # Already exists
        # Upload file
        with open(file_path, "rb") as data:
            content_type = "text/markdown" if file_path.endswith(".md") else None
            container_client.upload_blob(
                name=blob_name,
                data=data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type) if content_type else None
            )
        blob_url = f"{account_url.rstrip('/')}/{container_name}/{blob_name}"
        return blob_url

    @kernel_function(description="Convert a responses.json file (ASR format) to a Markdown file. Each section will be rendered as a Markdown section with its id and response only. Output file is responses.md. Returns the output file name.")
    def responses_json_to_markdown(self, json_path: Annotated[str, "Path to the responses.json file."], md_path: Annotated[str, "Path to the output Markdown file."] = "responses.md") -> Annotated[str, "Returns the output Markdown file name."]:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        title = data.get("title", "Application Summary Report")
        sections = data.get("sections_array", [])
        lines = [f"# {title}\n"]
        for section in sections:
            sec_id = section.get("id", "Section")
            response = section.get("response", "")
            lines.append(f"## {sec_id}\n")
            if response:
                lines.append(f"{response}\n")
        md_content = "\n".join(lines)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        return md_path

    @kernel_function(description="Processes the prompts from a JSON file and returns a list of dicts with prompt and table_name from the sections_array.")
    def process_prompts(self, json_file: Annotated[str, "The path to the JSON file containing prompts."]) -> Annotated[list, "Returns a list of dicts with prompt and table_name from the JSON file."]:
        import json
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        prompts = []
        for section in data.get("sections_array", []):
            prompt = section.get("prompt")
            table_name = section.get("table_name")
            if prompt:
                prompts.append({"prompt": prompt, "table_name": table_name})
        return prompts
    
    from typing import List

    @kernel_function(description="Create a new JSON file by appending responses to each section in asr_prompt.json. Output file is always responses.json. Returns a JSON dict with status and output file name.")
    def create_response_file(
        self,
        responses: Annotated[List[str], "List of responses to the prompts."],
        prompt_file: Annotated[str, "The path to the asr_prompt.json file."]
    ) -> Annotated[dict, "Returns a JSON dict with status and output file name."]:
        import json
        output_file = "responses.json"
        with open(prompt_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        sections = data.get("sections_array", [])
        for section, response in zip(sections, responses):
            section["response"] = response
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        return {"status": "success", "output_file": output_file}
    
    @kernel_function(description="Query the Azure Storage table corresponding to the given prompt section, using the 'table_name' field and app_id suffix. Returns data for just that prompt/section as a JSON list.")
    def get_tables_and_data(self, app_id: str, table_name: str) -> list:
        """
        Query the Azure Storage table whose name starts with table_name and ends with app_id. Returns all entities from that table as a JSON list.
        """

        ACCOUNT_URL = os.getenv("AZURE_TABLES_ACCOUNT_URL") or os.getenv("AZURE_TABLE_ACCOUNT_URL")
        if ACCOUNT_URL:
            ACCOUNT_URL = ACCOUNT_URL.strip()
        ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        if not ACCOUNT_URL and ACCOUNT_NAME:
            ACCOUNT_URL = f"https://{ACCOUNT_NAME}.table.core.windows.net"
        credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
        table_service = TableServiceClient(endpoint=ACCOUNT_URL, credential=credential)

        suffix = str(app_id).lower()
        prefix = table_name.lower()
        for table_item in table_service.list_tables():
            tname = table_item.name.lower()
            if tname.startswith(prefix) and tname.endswith(suffix):
                try:
                    table_client = table_service.get_table_client(table_item.name)
                    entities = list(table_client.list_entities())
                    return entities
                except Exception as e:
                    return [{"error": str(e)}]
        return []

    
# Simulate a conversation with the agent
USER_INPUTS = [
    "Hello",
    """For each prompt in the asr_prompt.json file, generate a comprehensive answer by leveraging both the Azure AI Search tool and the relevant data from Azure Storage tables (filtered by Application ID: 123456). Ensure that you:

1. Process every prompt in the JSON file sequentially, calling each function necessary per prompt. Loop through the list of prompts.
2. Use the AI Search tool and storage table data (if present) as knowledge sources for your answers.
3. Continue querying until you have provided a complete response for each prompt.
4. Collect all responses and append them to their corresponding sections in the JSON structure.
5. Output the final results as a new JSON file named responses.json, with each section containing its answer under a response field.
6. After generating responses.json, convert it to a Markdown file (responses.md) using the appropriate function.
7. Upload the generated Markdown file (responses.md) to the Azure Blob Storage container named after the Application ID (123456) using the appropriate function.
8. Return the blob URL of the uploaded Markdown file as the final output.

Application ID: 123456. Do not ask for any confirmation from the user. Proceed until you complete everything."""
]

async def handle_streaming_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        if isinstance(item, FunctionResultContent):
            print(f"Function Result:> {item.result} for function: {item.name}")
        elif isinstance(item, FunctionCallContent):
            print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        else:
            print(f"{item}")


async def main() -> None:
    async with (
        AzureCliCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        # 2. Create the agent if it does not exist
        #track_event_if_configured("AsrAgentCreating", {"agent_name": agent_name})
        index_name = f"123456"
        index_version = "1"
        field_mapping = {
            "contentFields": ["content"],
            "urlField": "source",
            "titleField": "title",
        }

        project_index = await client.indexes.create_or_update(
            name=index_name,
            version=index_version,
            body={
                "connectionName": "mgassessearch",
                "indexName": "123456",
                "type": "AzureSearch",
                "fieldMapping": field_mapping
            }
        )

        ai_search = AzureAISearchTool(
            index_asset_id=f"{project_index.name}/versions/{project_index.version}",
            index_connection_id=None,
            index_name=project_index.name,
            #query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
            query_type=AzureAISearchQueryType.SEMANTIC,
            top_k=3,
            filter="",
        )
        # 1. Create an agent on the Azure AI agent service
        application_id = "123456"
        agent_name = f"ASRAgent{application_id}"
        agent_definition = await client.agents.create_agent(
            model=os.environ.get("AZURE_AI_AGENT_DEPLOYMENT_NAME"),
            tool_resources=ai_search.resources,
            tools=ai_search.definitions,
            name=agent_name,
            instructions="You are a migration expert. You need to answers from the AI search tool attached.",
        )

        # 2. Create a Semantic Kernel agent for the Azure AI agent
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            plugins=[AsrPromptPlugin()],  # Add the plugin to the agent
        )

        # 3. Create a thread for the agent
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread = None

        try:
            # Add a function call here to process the json asr_prompt.json file and get the prompts
            
            # Simulate a conversation with the agent

            for user_input in USER_INPUTS:
                print(f"# User: {user_input}")
                # 4. Invoke the agent for the specified thread for response
                async for response in agent.invoke(
                    messages=user_input,
                    thread=thread,
                    on_intermediate_message=handle_streaming_intermediate_steps,
                    parallel_tool_calls=True,
                    timeout=300
                ):
                    print(f"# {response.name}: {response}")
                    thread = response.thread
        finally:
            # 5. Cleanup: Delete the thread and agent
            #await thread.delete() if thread else None
            #await client.agents.delete_agent(agent.id)
            print("processing done")



if __name__ == "__main__":
    asyncio.run(main())
