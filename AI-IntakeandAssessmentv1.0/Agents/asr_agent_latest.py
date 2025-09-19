# Copyright (c) Microsoft. All rights reserved.

import os
import asyncio
import shutil
import re
import json
from typing import Annotated, List
from dotenv import load_dotenv
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType
from azure.identity.aio import AzureCliCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, FunctionResultContent

load_dotenv()

# Standalone: Upload a file to Azure Blob Storage
def upload_file_to_container(file_path: str, app_id: str, blob_name: str = None) -> str:
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
# Standalone: Convert responses.json to Markdown
def responses_json_to_markdown(json_path: str, app_id: str, md_path: str = None) -> str:
    if md_path is None:
        md_path = f"responses-{app_id}.md"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    title = data.get("title", "Application Summary Report")
    sections = data.get("sections_array", [])
    lines = [f"# {title}\n"]
    for section in sections:
        sec_id = section.get("id", "Section")
        response = section.get("response", "")
        match = re.match(r"(\d+(?:\.\d+)*)(?:\s+)(.*)", sec_id)
        if match:
            numbering = match.group(1)
            heading_text = match.group(2)
            level = numbering.count('.') + 1
            heading = f"{'#' * level} {sec_id}"
        else:
            heading = f"## {sec_id}"
        lines.append(f"{heading}\n")
        if response:
            lines.append(f"{response}\n")
    md_content = "\n".join(lines)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    return md_path

# Standalone: Process prompts from JSON
def process_prompts(json_file: str) -> list:
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    prompts = []
    for section in data.get("sections_array", []):
        prompt = section.get("prompt")
        table_name = section.get("table_name")
        if prompt:
            prompts.append({"prompt": prompt, "table_name": table_name})
    return prompts

# Standalone: Create response file
def create_response_file(responses: list, prompt_file: str, app_id: str) -> dict:
    output_file = f"responses-{app_id}.json"
    with open(prompt_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    sections = data.get("sections_array", [])
    for section, response in zip(sections, responses):
        section["response"] = response
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)
    return {"status": "success", "output_file": output_file}





"""The ASR agent now relies solely on the unified search index that already contains
exported table rows (via JSONL snapshot) and uploaded documents. Direct table reads
have been removed to avoid duplication and reduce latency."""

async def handle_streaming_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        if isinstance(item, FunctionResultContent):
            print(f"Function Result:> {item.result} for function: {item.name}")
        elif isinstance(item, FunctionCallContent):
            print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        else:
            print(f"{item}")


async def run_asr_agent(application_id: str, client, thread=None) -> dict:
    """
    Run the ASR agent with the provided application ID and thread.
    
    Args:
        application_id: The application ID to process
        client: The Azure AI client
        thread: Optional thread to use (if None, a new one will be created)
    
    Returns:
        dict: Result containing status, output files, and blob URL
    """
    try:
        # 2. Create the agent if it does not exist
        #track_event_if_configured("AsrAgentCreating", {"agent_name": agent_name})
        index_name = f"{application_id}"
        index_version = "1"
        field_mapping = {
            "contentFields": ["content"],
            "urlField": "source",
            "titleField": "title",
        }

        project_index = await client.indexes.create_or_update(
            name=index_name,
            version=index_version,
            index={
                "connectionName": "mgassessearch",
                "indexName": index_name,
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
        agent_name = f"ASRAgent{application_id}"
        agent_definition = await client.agents.create_agent(
            model=os.environ.get("AZURE_AI_AGENT_DEPLOYMENT_NAME"),
            tool_resources=ai_search.resources,
            tools=ai_search.definitions,
            name=agent_name,
            instructions=(
                "You are a migration expert generating an Application Summary Report. "
                "Use ONLY the attached Azure AI Search tool for answers. The index already contains "
                "(a) uploaded source documents and (b) exported application tables flattened into JSONL with fields like _SourceTable and Key. "
                "Do not request external knowledge or fabricate data. For each prompt, perform focused search queries; if information is missing, reply exactly: 'No relevant information found in the provided data.' "
                "Cite sources briefly if available (document name or _SourceTable + RowKey)."
            ),
        )

        # 2. Create a Semantic Kernel agent for the Azure AI agent
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            #plugins=[AsrPromptPlugin()],  # Add the plugin to the agent
        )

        # 3. Use provided thread or create new one
        # If no thread is provided, a new thread will be
        # created and returned with the initial response

        import json
        prompt_file = "asr_prompt.json"
        app_id = application_id
        
        # Check if prompt file exists
        if not os.path.exists(prompt_file):
            return {"status": "error", "message": f"Prompt file {prompt_file} not found"}
        
        # Load prompts and table names
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sections = data.get("sections_array", [])
        responses = []
        
        for section in sections:
            prompt = section.get("prompt", "")
            # table_name field no longer used; retained in JSON for backward compatibility
            if not prompt:
                responses.append("")
                continue
            # Augment prompt with explicit guidance to search (idempotent, avoids duplicating if already added)
            guidance_suffix = (
                "\n\nInstructions: Use the search tool to retrieve only relevant facts. "
                "Search iteratively if needed. Base all content strictly on retrieved passages. "
                "If nothing is found, respond with: No relevant information found in the provided data."
            )
            if guidance_suffix not in prompt:
                user_message = prompt + guidance_suffix
            else:
                user_message = prompt
            print(f"# User: {user_message}")
            # Invoke the agent for this prompt only
            async for response in agent.invoke(
                messages=user_message,
                thread=thread,
                on_intermediate_message=handle_streaming_intermediate_steps,
                parallel_tool_calls=False,
                timeout=120
            ):
                print(f"# {response.name}: {response}")
                thread = response.thread
                responses.append(str(response))
                break  # Only one response per prompt
        
        # Save responses to responses-{app_id}.json
        result = create_response_file(responses, prompt_file, app_id)
        if result.get("status") == "success":
            responses_json_path = result.get("output_file")
            md_path = responses_json_to_markdown(responses_json_path, app_id)
            print(f"Markdown file created: {md_path}")

            # Upload to blob storage, handle versioning if file exists
            version = 1
            base_md_path = md_path
            file_name = os.path.basename(md_path)
            file_root, file_ext = os.path.splitext(file_name)
            container_files = []
            try:
                from azure.storage.blob import BlobServiceClient
                account_url = os.getenv("AZURE_BLOB_ACCOUNT_URL") or os.getenv("AZURE_STORAGE_ACCOUNT_URL") or os.getenv("AZURE_TABLES_ACCOUNT_URL") or os.getenv("AZURE_TABLE_ACCOUNT_URL")
                if account_url:
                    account_url = account_url.strip()
                account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
                if not account_url and account_name:
                    account_url = f"https://{account_name}.blob.core.windows.net"
                credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
                blob_service = BlobServiceClient(account_url, credential=credential)
                container_name = str(app_id).lower()
                container_client = blob_service.get_container_client(container_name)
                # List blobs in the container
                container_files = [b.name for b in container_client.list_blobs()]
            except Exception:
                pass
            new_file_name = file_name
            while new_file_name in container_files:
                version += 1
                new_file_name = f"{file_root}_v{version}{file_ext}"
            if new_file_name != file_name:
                # Copy to new versioned file
                new_md_path = os.path.join(os.path.dirname(md_path), new_file_name)
                import shutil
                shutil.copy(md_path, new_md_path)
                md_path = new_md_path
            blob_url = upload_file_to_container(md_path, app_id, new_file_name)
            print(f"Markdown file uploaded to blob storage: {blob_url}")
            
            return {
                "status": "success", 
                "agent_id": agent_definition.id,
                "thread": thread,
                "output_file": responses_json_path,
                "markdown_file": md_path,
                "blob_url": blob_url
            }
        else:
            return {"status": "error", "message": "Failed to create response file"}
            
    except Exception as e:
        print(f"Error in ASR agent execution: {str(e)}")
        return {"status": "error", "message": str(e)}


async def main() -> None:
    """Main function for standalone execution"""
    async with (
        AzureCliCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        application_id = "2007"
        result = await run_asr_agent(application_id, client)
        print(f"ASR Agent execution result: {result}")
        print("processing done")



if __name__ == "__main__":
    asyncio.run(main())
