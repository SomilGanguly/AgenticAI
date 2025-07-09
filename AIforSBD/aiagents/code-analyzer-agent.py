import os
import asyncio
import logging
import subprocess
import json
import math
from typing import List, Dict, Any
from semantic_kernel.contents.annotation_content import AnnotationContent
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from azure.ai.agents.models import CodeInterpreterTool, FilePurpose
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.contents.utils.author_role import AuthorRole
from datetime import datetime
from dotenv import load_dotenv
from datetime import timedelta
load_dotenv()

logging.basicConfig(
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("kernel").setLevel(logging.DEBUG)

def split_json_by_resources(json_file_path: str, output_dir: str, max_resources_per_chunk: int = 10) -> List[str]:
    """
    Split a large Terraform JSON file into smaller chunks based on resource count.
    Returns list of chunk file paths.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract resources from planned_values or resource_changes
    resources = []
    if 'planned_values' in data and 'root_module' in data['planned_values']:
        resources.extend(data['planned_values']['root_module'].get('resources', []))
        # Also check child modules
        if 'child_modules' in data['planned_values']['root_module']:
            for module in data['planned_values']['root_module']['child_modules']:
                resources.extend(module.get('resources', []))
    
    if 'resource_changes' in data:
        resource_changes = data['resource_changes']
    else:
        resource_changes = []
    
    if not resources and not resource_changes:
        # If no resources found, return original file
        return [json_file_path]
    
    # Calculate number of chunks needed
    total_items = max(len(resources), len(resource_changes))
    num_chunks = math.ceil(total_items / max_resources_per_chunk)
    
    chunk_files = []
    
    for i in range(num_chunks):
        start_idx = i * max_resources_per_chunk
        end_idx = min((i + 1) * max_resources_per_chunk, total_items)
        
        # Create chunk data structure
        chunk_data = {
            "format_version": data.get("format_version"),
            "terraform_version": data.get("terraform_version"),
            "variables": data.get("variables", {}),
            "configuration": data.get("configuration", {}),
            "chunk_info": {
                "chunk_number": i + 1,
                "total_chunks": num_chunks,
                "resources_range": f"{start_idx}-{end_idx-1}"
            }
        }
        
        # Add subset of resources
        if resources:
            chunk_resources = resources[start_idx:end_idx]
            chunk_data["planned_values"] = {
                "root_module": {
                    "resources": chunk_resources
                }
            }
        
        if resource_changes:
            chunk_resource_changes = resource_changes[start_idx:end_idx]
            chunk_data["resource_changes"] = chunk_resource_changes
        
        # Save chunk
        chunk_filename = f"terraform_chunk_{i+1}_of_{num_chunks}.json"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2)
        
        chunk_files.append(chunk_path)
        print(f"Created chunk {i+1}/{num_chunks}: {chunk_filename}")
    
    return chunk_files

def create_summary_json(json_file_path: str, output_path: str) -> str:
    """
    Create a summary JSON with high-level information about the Terraform plan.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = {
        "format_version": data.get("format_version"),
        "terraform_version": data.get("terraform_version"),
        "variables": data.get("variables", {}),
        "configuration": {
            "provider_config": data.get("configuration", {}).get("provider_config", {}),
            "root_module": {
                "resources": []
            }
        },
        "summary": {
            "total_resources": 0,
            "resource_types": {},
            "provider_summary": {}
        }
    }
    
    # Count resources and types
    all_resources = []
    
    if 'planned_values' in data and 'root_module' in data['planned_values']:
        all_resources.extend(data['planned_values']['root_module'].get('resources', []))
        if 'child_modules' in data['planned_values']['root_module']:
            for module in data['planned_values']['root_module']['child_modules']:
                all_resources.extend(module.get('resources', []))
    
    if 'resource_changes' in data:
        for change in data['resource_changes']:
            if change not in all_resources:
                all_resources.append(change)
    
    # Create summary statistics
    resource_types = {}
    for resource in all_resources:
        resource_type = resource.get('type', 'unknown')
        if resource_type not in resource_types:
            resource_types[resource_type] = 0
        resource_types[resource_type] += 1
        
        # Add basic resource info without sensitive data
        summary["configuration"]["root_module"]["resources"].append({
            "type": resource.get("type"),
            "name": resource.get("name"),
            "provider": resource.get("provider_name"),
            "mode": resource.get("mode", "managed")
        })
    
    summary["summary"]["total_resources"] = len(all_resources)
    summary["summary"]["resource_types"] = resource_types
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return output_path

def getAllFiles(dir: str) -> str:
    return ",".join(os.listdir(dir))

async def process_json_chunks(agent, client, chunk_files: List[str], analysis_prompt: str, thread=None) -> List[str]:
    """
    Process multiple JSON chunks and return analysis results.
    """
    results = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
        x = await client.agents.files.clear
        # Upload chunk file
        file = await client.agents.files.upload_and_poll(file_path=chunk_file, purpose=FilePurpose.AGENTS)
        #client.agents.files.upload_and_poll
        # Update code interpreter with this chunk
        code_interpreter = CodeInterpreterTool(file_ids=[file.id])
        
        # Update agent
        agent_definition = await client.agents.update_agent(
            agent_id=agent.id,
            tools=code_interpreter.definitions,
            tool_resources=code_interpreter.resources,
        )
        
        # Process this chunk
        chunk_prompt = f"{analysis_prompt} (Processing chunk {i+1} of {len(chunk_files)})"
        
        chunk_results = []
        async for response in agent.invoke(messages=chunk_prompt, thread=thread):
            if response.role != AuthorRole.TOOL:
                chunk_results.append(str(response))
                thread = response.thread
                
                if len(response.items) > 0:
                    for item in response.items:
                        if isinstance(item, AnnotationContent):
                            response_content = await client.agents.get_file_content(file_id=item.file_id)
                            content_bytes = bytearray()
                            async for chunk_content in response_content:
                                content_bytes.extend(chunk_content)
                            tab_delimited_text = content_bytes.decode("utf-8")
                            chunk_results.append(tab_delimited_text)
        
        results.extend(chunk_results)
        
        # Clean up the chunk file from agent (optional, to manage memory)
        try:
            await client.agents.delete_file(file.id)
        except:
            pass  # Continue if deletion fails
    
    return results

async def main():
    ai_agent_settings = AzureAIAgentSettings(
        # model_deployment_name="gpt-4o",
        # project_connection_string=os.getenv("PROJECT_CONNECTION_STRING"),
        # endpoint=os.getenv("AGENT_ENDPOINT")
    )
    
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(
            credential=creds,
            #conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
        ) as client,
    ):
        #current_dir = os.path.dirname(os.path.realpath(__file__))
        current_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        terraform_dir = os.path.join(current_dir, "vending-pattern")
        baseline_dir = os.path.join(current_dir, "baselines")
        
        if not os.path.exists(terraform_dir):
            raise FileNotFoundError(f"Terraform files directory not found: {terraform_dir}")

        terraform_file_path = os.path.join(terraform_dir, 'terraform_file.json')
        
        if not os.path.exists(terraform_file_path):
            raise FileNotFoundError(f"terraform_file.json not found at: {terraform_file_path}")
        
        # Check file size and determine processing strategy
        file_size_mb = os.path.getsize(terraform_file_path) / (1024 * 1024)
        print(f"Terraform JSON file size: {file_size_mb:.2f} MB")
        
        # Create chunks directory
        chunks_dir = os.path.join(current_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Strategy 1: For very large files (>5MB), split into chunks
        # if file_size_mb > 5:
        print("Large file detected. Splitting into chunks...")
        chunk_files = split_json_by_resources(terraform_file_path, chunks_dir, max_resources_per_chunk=5)
        summary_path = os.path.join(chunks_dir, "terraform_summary.json")
        create_summary_json(terraform_file_path, summary_path)
        terraform_files = [summary_path] + chunk_files
        # else:
        #     # For smaller files, process normally
        #     terraform_files = [terraform_file_path]
        
        # Create initial agent
        uploaded_files = []
        #files = await client.agents.files. (purpose=FilePurpose.AGENTS)
        for tf_file in terraform_files[:1]:  # Start with summary or original file
            file = await client.agents.files.upload_and_poll(file_path=tf_file, purpose=FilePurpose.AGENTS)
            uploaded_files.append(file)
 
        code_interpreter = CodeInterpreterTool(file_ids = [file_info.id for file_info in uploaded_files])

        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="code-analyzer-agent",
            tools=code_interpreter.definitions,
            tool_resources=code_interpreter.resources,
        )

        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
        )
        agent.polling_options.run_polling_timeout = timedelta(minutes=25)

        thread: AzureAIAgentThread = None
        baselines = getAllFiles('baselines')
        print(f"Available baseline files: {baselines}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        markdown_path = os.path.join(current_dir, "outputs", f"security_evaluation_report_{timestamp}.md")
        
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write("# Security Evaluation Report\n\n")
            md_file.write(f"_Analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}_\n\n")
            
            try:
                # Process overview first
                if file_size_mb > 5:
                    overview_prompt = "Analyze the provided Terraform summary JSON file to understand the overall infrastructure being deployed. Provide a high-level overview of the resources, their types, and general security considerations."
                    
                    async for response in agent.invoke(messages=overview_prompt, thread=thread):
                        if response.role != AuthorRole.TOOL:
                            print(f"# Agent Overview: {response}")
                            md_file.write(f"## Infrastructure Overview\n\n{response}\n\n")
                            thread = response.thread
                    
                    # Process detailed chunks
                    if len(terraform_files) > 1:
                        md_file.write("## Detailed Security Analysis by Resource Groups\n\n")
                        
                        detailed_prompt = "Analyze the provided JSON chunk for security vulnerabilities and misconfigurations. Focus on the specific resources in this chunk and provide detailed security recommendations."
                        
                        chunk_results = await process_json_chunks(
                            agent, client, terraform_files[1:], detailed_prompt, thread
                        )
                        
                        for i, result in enumerate(chunk_results):
                            md_file.write(f"### Chunk {i+1} Analysis\n\n{result}\n\n")
                else:
                    # Process single file normally
                    user_inputs = [
                        "Breakdown the provided JSON file for better analysis and understanding in the next steps.",
                        "Analyze the provided JSON file, which contains the Terraform plan output. Please perform a comprehensive security analysis of all Azure resources defined in this configuration. For each resource, examine the properties, configurations, and values being set, then identify existing security measures and highlight any missing or inadequate security controls that should be implemented.",
                    ]
                    
                    for user_input in user_inputs:
                        print(f"# User: '{user_input}'")
                        async for response in agent.invoke(messages=user_input, thread=thread):
                            if response.role != AuthorRole.TOOL:
                                print(f"# Agent: {response}")
                                md_file.write(f"{response}\n\n")
                                
                                if len(response.items) > 0:
                                    for item in response.items:
                                        if isinstance(item, AnnotationContent):
                                            response_content = await client.agents.get_file_content(file_id=item.file_id)
                                            content_bytes = bytearray()
                                            async for chunk in response_content:
                                                content_bytes.extend(chunk)
                                            tab_delimited_text = content_bytes.decode("utf-8")
                                            print(tab_delimited_text)
                                
                                thread = response.thread

                # Get baseline recommendations
                baseline_prompt = f"Based on the resources being used, which one of these baseline files should be used for comparison: {baselines}, provide exact names of the files as comma separated values in a single line, don't include anything other than the names in response."
                print(f"# User: '{baseline_prompt}'")
                
                async for response in agent.invoke(messages=baseline_prompt, thread=thread):
                    if response.role != AuthorRole.TOOL:
                        print(f"# Agent: {response}")
                        thread = response.thread
                        baseline_files_to_compare = str(response).split(",")
                        
                        if len(baseline_files_to_compare) > 0:
                            baseline_uploaded_files = []
                            for baseline_file in baseline_files_to_compare:
                                baseline_file_path = os.path.join(baseline_dir, baseline_file.strip())
                                if os.path.exists(baseline_file_path):
                                    file = await client.agents.files.upload_and_poll(file_path=baseline_file_path, purpose=FilePurpose.AGENTS)
                                    baseline_uploaded_files.append(file)

                            # Update code interpreter with baseline files
                            if baseline_uploaded_files:
                                all_file_ids = [file.id for file in uploaded_files] + [file.id for file in baseline_uploaded_files]
                                code_interpreter = CodeInterpreterTool(file_ids=all_file_ids)

                                # Update the agent with the new code interpreter
                                agent_definition = await client.agents.update_agent(
                                    agent_id=agent.id,
                                    tools=code_interpreter.definitions,
                                    tool_resources=code_interpreter.resources,
                                )

                        md_file.write("## Security Implementation Checklist\n\n")
                        final_prompt = "For each resource separately compare the security measures from your analysis and baseline files and provide a table of security measures that are already present, missing, and need to be implemented. The table should have the following columns: 'Security Measure', 'Present', 'Missing', 'Needs Implementation'. Provide response in markdown format and also include terraform attributes that need to be added for the recommendations.\n\n"
                        print(f"# User: '{final_prompt}'")
                            
                        async for response in agent.invoke(messages=final_prompt, thread=thread):
                            if response.role != AuthorRole.TOOL:
                                print(f"# Agent: {response}")
                                md_file.write(f"{response}\n\n")
                                thread = response.thread
    
            finally:
                # Cleanup: Delete the thread and agent
                # if thread:
                #     await thread.delete()
                # await client.agents.delete_agent(agent.id)
                
                # Clean up chunk files
                if file_size_mb > 5:
                    try:
                        import shutil
                        shutil.rmtree(chunks_dir)
                        print("Cleaned up temporary chunk files")
                    except:
                        print("Could not clean up chunk files")
                
            print("Analysis completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())