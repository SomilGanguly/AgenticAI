import os
import asyncio
import logging
import json
import math
from typing import List, Dict, Any
from semantic_kernel.contents.annotation_content import AnnotationContent
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.ai.agents.models import CodeInterpreterTool, FilePurpose, AzureAISearchTool, AzureAISearchQueryType
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.identity import DefaultAzureCredential 
from semantic_kernel.contents.utils.author_role import AuthorRole
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tools.github_fetcher import GitHubFetcher
from tools.terraform_plan_generator import TerraformPlanGenerator

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
        
        # Upload chunk file
        file = await client.agents.files.upload_and_poll(file_path=chunk_file, purpose=FilePurpose.AGENTS)
        
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
    # Load environment variables
    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
    SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
    AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT")
    
    # Validate environment variables
    if not all([SEARCH_ENDPOINT, SEARCH_INDEX_NAME, SEARCH_API_KEY]):
        print("Error: Missing required Azure Search environment variables")
        print("Please ensure the following are set in your .env file:")
        print("  - AZURE_SEARCH_ENDPOINT")
        print("  - AZURE_SEARCH_INDEX_NAME")
        print("  - AZURE_SEARCH_API_KEY")
        return
    
    ai_agent_settings = AzureAIAgentSettings(
        model_deployment_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        endpoint=os.getenv("AGENT_ENDPOINT")
    )
    
    if not ai_agent_settings.endpoint:
        print("Error: AGENT_ENDPOINT not set in environment variables")
        return
    
    # Create synchronous credential for AIProjectClient
    sync_credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    
    async with (
        AsyncDefaultAzureCredential() as async_creds,
        AzureAIAgent.create_client(
            credential=async_creds,
            endpoint=ai_agent_settings.endpoint,
        ) as client,
    ):
        # Initialize tools
        github_fetcher = GitHubFetcher()
        tf_generator = TerraformPlanGenerator()
        
        agent = None
        thread = None
        
        try:
            # Get GitHub URL from user input
            github_url = input("Enter GitHub repository URL: ").strip()
            
            if not github_url:
                print("Error: No URL provided")
                return
            
            print("\n=== Step 1: Fetching GitHub Repository ===")
            fetch_result = github_fetcher.fetch_repo(github_url)
            
            if fetch_result["status"] != "success":
                print(f"Error: {fetch_result['message']}")
                return
            
            repo_path = fetch_result["repo_path"]
            print(f"Repository cloned to: {repo_path}")
            print(f"Found {len(fetch_result['terraform_files'])} Terraform files")
            
            if not fetch_result['terraform_files']:
                print("No Terraform files found in the repository.")
                return
            
            print("\n=== Step 2: Generating Terraform Plan ===")
            # Find directories with .tf files
            tf_dirs = list({os.path.dirname(f) for f in fetch_result["terraform_files"] if f.endswith('.tf')})
            
            if not tf_dirs:
                print("No Terraform directories found.")
                return
            
            # Filter out test and example directories
            valid_tf_dirs = [d for d in tf_dirs if not any(skip in d.lower() for skip in ['test', 'example', '.terraform', 'module'])]
            
            # If no valid directories after filtering, use all directories but warn
            if not valid_tf_dirs:
                print("Warning: Only test/example/module directories found. Using all directories...")
                valid_tf_dirs = tf_dirs
            
            # Select the appropriate directory
            if len(valid_tf_dirs) > 1:
                print(f"\nFound {len(valid_tf_dirs)} directories with Terraform files:")
                for i, dir_path in enumerate(valid_tf_dirs):
                    tf_count = len([f for f in fetch_result["terraform_files"] if os.path.dirname(f) == dir_path and f.endswith('.tf')])
                    rel_path = os.path.relpath(dir_path, repo_path)
                    print(f"{i+1}. {rel_path} ({tf_count} .tf files)")
                
                # Ask user to select
                while True:
                    try:
                        choice = input(f"\nSelect directory (1-{len(valid_tf_dirs)}) or press Enter for auto-select: ").strip()
                        if choice == "":
                            # Auto-select the directory with most .tf files
                            main_tf_dir = max(valid_tf_dirs, key=lambda d: len([f for f in fetch_result["terraform_files"] if os.path.dirname(f) == d and f.endswith('.tf')]))
                            print(f"Auto-selected: {os.path.relpath(main_tf_dir, repo_path)}")
                            break
                        else:
                            idx = int(choice) - 1
                            if 0 <= idx < len(valid_tf_dirs):
                                main_tf_dir = valid_tf_dirs[idx]
                                print(f"Selected: {os.path.relpath(main_tf_dir, repo_path)}")
                                break
                            else:
                                print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            else:
                main_tf_dir = valid_tf_dirs[0]
                print(f"Using directory: {os.path.relpath(main_tf_dir, repo_path)}")
            
            # Find .tfvars file in the selected directory
            tfvars_file = None
            for tf_file in fetch_result["terraform_files"]:
                if tf_file.endswith('.tfvars') and os.path.dirname(tf_file) == main_tf_dir:
                    tfvars_file = tf_file
                    print(f"Found .tfvars file: {os.path.basename(tfvars_file)}")
                    break
            
            # Generate Terraform plan
            print("\nGenerating Terraform plan (this may take a moment)...")
            plan_result = tf_generator.generate_plan(main_tf_dir, tfvars_file)
            
            if plan_result["status"] != "success":
                print(f"Error: {plan_result['message']}")
                
                # Offer to try another directory
                if len(valid_tf_dirs) > 1:
                    retry = input("\nWould you like to try another directory? (y/n): ").strip().lower()
                    if retry == 'y':
                        # Remove the failed directory and restart selection
                        valid_tf_dirs.remove(main_tf_dir)
                        # Recursively call with remaining directories
                        # (In a real implementation, you'd restructure this as a loop)
                        print("Please restart and select a different directory.")
                return
            
            terraform_file_path = plan_result["json_file"]
            print(f"Terraform JSON generated: {terraform_file_path}")
            
            # Check file size and create chunks if needed
            file_size_mb = os.path.getsize(terraform_file_path) / (1024 * 1024)
            print(f"Terraform JSON file size: {file_size_mb:.2f} MB")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            chunks_dir = os.path.join(current_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            
            if file_size_mb > 5:
                print("Large file detected. Splitting into chunks...")
                chunk_files = split_json_by_resources(terraform_file_path, chunks_dir, max_resources_per_chunk=5)
                summary_path = os.path.join(chunks_dir, "terraform_summary.json")
                create_summary_json(terraform_file_path, summary_path)
                terraform_files = [summary_path] + chunk_files
            else:
                terraform_files = [terraform_file_path]
            
            # Create Azure AI Search tool
            print("\n=== Step 3: Setting up Azure AI Search ===")
            
            # Use synchronous credential for AIProjectClient
            project_client = AIProjectClient(
                endpoint=AGENT_ENDPOINT,
                credential=sync_credential
            )

            # Azure AI Search setup
            try:
                azure_ai_conn_id = project_client.connections.get_default(ConnectionType.AZURE_AI_SEARCH).id
                print(f"Found Azure AI Search connection:")
            except Exception as e:
                print(f"Warning: Could not find default Azure AI Search connection: {e}")
                print("Proceeding without Azure AI Search integration...")
                
                # Create agent without search tool
                print("Uploading Terraform configuration to Azure AI Agent...")
                uploaded_files = []
                for tf_file in terraform_files[:1]:
                    file = await client.agents.files.upload_and_poll(file_path=tf_file, purpose=FilePurpose.AGENTS)
                    uploaded_files.append(file)
                
                code_interpreter = CodeInterpreterTool(file_ids=[file_info.id for file_info in uploaded_files])
                
                print("Creating security analysis agent (without search)...")
                agent_definition = await client.agents.create_agent(
                    model=ai_agent_settings.model_deployment_name,
                    name="security-analyzer-agent",
                    instructions="""You are a security analysis agent that analyzes Terraform configurations 
                    for security issues. Analyze the provided Terraform configuration and identify security gaps, 
                    misconfigurations, and non-compliant resources based on general security best practices.
                    
                    When analyzing:
                    1. Check for encryption at rest and in transit
                    2. Verify network security configurations
                    3. Check for proper access controls and authentication
                    4. Identify any hardcoded secrets or sensitive data
                    5. Verify backup and disaster recovery configurations
                    
                    Return results in JSON format with the following structure:
                    {
                        "security_gaps": [
                            {
                                "resource_type": "string",
                                "resource_name": "string",
                                "security_issue": "string",
                                "severity": "high|medium|low",
                                "baseline_reference": "string",
                                "recommendation": "string",
                                "terraform_fix": "string"
                            }
                        ],
                        "summary": {
                            "total_issues": 0,
                            "high_severity": 0,
                            "medium_severity": 0,
                            "low_severity": 0
                        }
                    }
                    """,
                    tools=code_interpreter.definitions,
                    tool_resources=code_interpreter.resources
                )
            else:
                # Create the search tool with the correct structure
                search_tool = AzureAISearchTool(
                    index_connection_id=azure_ai_conn_id,
                    index_name=SEARCH_INDEX_NAME,
                    query_type=AzureAISearchQueryType.SIMPLE,
                    top_k=20
                )
                
                # Upload files and create agent
                print("Uploading Terraform configuration to Azure AI Agent...")
                uploaded_files = []
                for tf_file in terraform_files[:1]:  # Start with summary or original file
                    file = await client.agents.files.upload_and_poll(file_path=tf_file, purpose=FilePurpose.AGENTS)
                    uploaded_files.append(file)
                
                code_interpreter = CodeInterpreterTool(file_ids=[file_info.id for file_info in uploaded_files])
                
                # Combine tools
                all_tools = [*code_interpreter.definitions, *search_tool.definitions]
                all_resources = {**code_interpreter.resources, **search_tool.resources}
                
                # Create agent with both tools
                print("Creating security analysis agent...")
                agent_definition = await client.agents.create_agent(
                    model=ai_agent_settings.model_deployment_name,
                    name="security-analyzer-agent",
                    instructions="""You are a security analysis agent that analyzes Terraform configurations 
                    against security baselines. Use the Azure AI Search tool to find relevant security 
                    standards and baselines, then compare them with the provided Terraform configuration.
                    
                    When analyzing:
                    1. First search for relevant security baselines using queries like "azure security baseline", 
                       "terraform security", "network security", "storage security", etc.
                    2. Analyze the Terraform configuration for security issues
                    3. Compare against the baselines found in the search
                    4. Return results in JSON format with the following structure:
                       {
                         "security_gaps": [
                           {
                             "resource_type": "string",
                             "resource_name": "string",
                             "security_issue": "string",
                             "severity": "high|medium|low",
                             "baseline_reference": "string",
                             "recommendation": "string",
                             "terraform_fix": "string"
                           }
                         ],
                         "summary": {
                           "total_issues": 0,
                           "high_severity": 0,
                           "medium_severity": 0,
                           "low_severity": 0
                         }
                       }
                    """,
                    tools=all_tools,
                    tool_resources=all_resources
                )
            
            agent = AzureAIAgent(
                client=client,
                definition=agent_definition,
            )
            agent.polling_options.run_polling_timeout = timedelta(minutes=25)
            
            print("\n=== Step 4: Analyzing Security ===")
            print("Starting security analysis (this may take several minutes)...")
            
            # Create output directory
            outputs_dir = os.path.join(current_dir, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_output_path = os.path.join(outputs_dir, f"security_analysis_{timestamp}.json")
            
            analysis_prompt = """Analyze the provided Terraform configuration for security issues. 
            Search the knowledge base for relevant Azure security baselines and best practices. 
            Compare the configuration against these baselines and identify all security gaps, 
            misconfigurations, and non-compliant resources. Return the results in the specified JSON format."""
            
            # If we have multiple chunks, process them
            if len(terraform_files) > 1 and file_size_mb > 5:
                print(f"Processing {len(chunk_files)} chunks...")
                results = await process_json_chunks(agent, client, chunk_files, analysis_prompt, thread)
                # Combine results and save
                combined_results = "\n".join(results)
                with open(json_output_path.replace('.json', '_raw.txt'), 'w', encoding='utf-8') as f:
                    f.write(combined_results)
            else:
                # Process single file
                async for response in agent.invoke(messages=analysis_prompt, thread=thread):
                    if response.role != AuthorRole.TOOL:
                        print(f"\nAgent response received...")
                        
                        # Try to extract JSON from response
                        try:
                            response_text = str(response)
                            # Find JSON content
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            
                            if json_start != -1 and json_end != 0:
                                json_content = response_text[json_start:json_end]
                                security_analysis = json.loads(json_content)
                                
                                # Save to file
                                with open(json_output_path, 'w', encoding='utf-8') as f:
                                    json.dump(security_analysis, f, indent=2)
                                
                                print(f"\n✓ Security analysis saved to: {json_output_path}")
                                print(f"\n=== Analysis Summary ===")
                                print(f"Total issues found: {security_analysis['summary']['total_issues']}")
                                print(f"  - High severity: {security_analysis['summary']['high_severity']}")
                                print(f"  - Medium severity: {security_analysis['summary']['medium_severity']}")
                                print(f"  - Low severity: {security_analysis['summary']['low_severity']}")
                                
                                # Show a few examples if available
                                if security_analysis['security_gaps']:
                                    print(f"\nTop security issues:")
                                    for i, gap in enumerate(security_analysis['security_gaps'][:3]):
                                        print(f"\n{i+1}. {gap['security_issue']}")
                                        print(f"   Resource: {gap['resource_type']} - {gap['resource_name']}")
                                        print(f"   Severity: {gap['severity']}")
                                        print(f"   Recommendation: {gap['recommendation']}")
                        except Exception as e:
                            print(f"Could not parse JSON response: {e}")
                            # Save raw response
                            with open(json_output_path.replace('.json', '_raw.txt'), 'w', encoding='utf-8') as f:
                                f.write(str(response))
                            print(f"Raw response saved to: {json_output_path.replace('.json', '_raw.txt')}")
                        
                        thread = response.thread
            
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted by user.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            logging.exception("Unexpected error in main")
        finally:
            # Cleanup
            print("\n=== Cleaning up ===")
            
            try:
                github_fetcher.cleanup()
            except Exception as e:
                print(f"Warning: Error during GitHub cleanup: {e}")
            
            if thread:
                try:
                    await thread.delete()
                except Exception as e:
                    print(f"Warning: Error deleting thread: {e}")
            
            if agent and hasattr(agent, 'id'):
                try:
                    await client.agents.delete_agent(agent.id)
                except Exception as e:
                    print(f"Warning: Error deleting agent: {e}")
            
            print("\n✓ Analysis completed!")

if __name__ == "__main__":
    asyncio.run(main())