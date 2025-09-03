import os
import asyncio
import logging
import subprocess
from semantic_kernel.contents.annotation_content import AnnotationContent
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from azure.ai.projects.models import CodeInterpreterTool, FilePurpose
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

# def create_terraform_json(terraform_dir : str) -> None: 
#     subprocess.run(
#         ["terraform", "show", "-json","tf.plan",  ">", "terraform_file.json"],  
#         shell=True, 
#         check=True,
#         cwd=terraform_dir,

#     )
#     print(f"Generated terraform_file.json in {terraform_dir}")

def getAllFiles(dir : str)->str:
    return ",".join(os.listdir(dir))


async def main():
    
    ai_agent_settings = AzureAIAgentSettings(
        model_deployment_name="gpt-4o",
        project_connection_string=os.getenv("PROJECT_CONNECTION_STRING"),
        endpoint=os.getenv("AGENT_ENDPOINT")
    )
    async with (
        DefaultAzureCredential(
        ) as creds,
        AzureAIAgent.create_client(
            credential=creds,
            conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
        ) as client,
    ):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        terraform_dir = os.path.join(
            current_dir,
            "terraform_files" 
        )
        baseline_dir = os.path.join(
            current_dir,
            "baselines"
        )
        if not os.path.exists(terraform_dir):
            raise FileNotFoundError(f"Terraform files directory not found: {terraform_dir}")
  
        # try:
        #     create_terraform_json(terraform_dir)
        # except subprocess.CalledProcessError as e:
        #     raise RuntimeError(f"Error running terraform command: {e}")
        

        terraform_file_path = os.path.join(terraform_dir, 'terraform_file.json')
        
        if not os.path.exists(terraform_file_path):
            raise FileNotFoundError(f"terraform_file.json not found at: {terraform_file_path}")
            
        terraform_files = [terraform_file_path]
        
        uploaded_files = []
        for tf_file in terraform_files:
            file = await client.agents.upload_file_and_poll(file_path=tf_file, purpose=FilePurpose.AGENTS)
            uploaded_files.append(file)
 
        code_interpreter = CodeInterpreterTool(file_ids=[file.id for file in uploaded_files])

        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
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
        user_inputs = [
            "breakdown the provided json file for better analysis and understanding in the next steps."
            "Analyze the provided JSON file, which contains the Terraform plan output. Please perform a comprehensive security analysis of all Azure resources defined in this configuration. For each resource, examine the properties, configurations, and values being set, then identify existing security measures and highlight any missing or inadequate security controls that should be implemented.",
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        markdown_path = os.path.join(current_dir, "outputs" , f"security_recommendations_{timestamp}.md")
        
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write("# Terraform Security Recommendations\n\n")
            md_file.write(f"_Analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}_\n\n")
            try:
                for user_input in user_inputs:
                    print(f"# User: '{user_input}'")
                    async for response in agent.invoke(messages=user_input, thread=thread):
                        if response.role != AuthorRole.TOOL:
                            print(f"# Agent: {response}")
                            if len(response.items) > 0:
                                for item in response.items:
                                    if isinstance(item, AnnotationContent):
                                        print(f"\n`{item.quote}` => {item.file_id}")
                                        response_content = await client.agents.get_file_content(file_id=item.file_id)
                                        content_bytes = bytearray()
                                        async for chunk in response_content:
                                            content_bytes.extend(chunk)
                                        tab_delimited_text = content_bytes.decode("utf-8")
                                        print(f"# Agent: {response}")
                                        md_file.write(f"{response}\n\n")
                                        print(tab_delimited_text)
                        thread = response.thread

                baseline_prompt = f"Based on the resources being used, which one of these baselines files should be used for comparison: {baselines}, provide exact names of the files as comma seperated values in a single line, don't include anything other than the names in response."
                print(f"# User: '{baseline_prompt}'")
                async for response in agent.invoke(messages=baseline_prompt, thread=thread):
                    if response.role != AuthorRole.TOOL:
                        print(f"# Agent: {response}")
                        thread = response.thread
                        baseline_files_to_compare = str(response).split(",")
                        if(len(baseline_files_to_compare) == 0):
                            baseline_uploaded_files = []
                            for baseline_file in baseline_files_to_compare:
                                baseline_file_path = os.path.join(baseline_dir, baseline_file.strip())
                                if not os.path.exists(baseline_file_path):
                                    raise FileNotFoundError(f"Baseline file not found: {baseline_file_path}")
                                baseline_file_path = os.path.join(baseline_dir, baseline_file.strip())
                                file = await client.agents.upload_file_and_poll(file_path=baseline_file_path, purpose=FilePurpose.AGENTS)
                                baseline_uploaded_files.append(file)

                                # Update code interpreter with baseline files
                            all_file_ids = [file.id for file in uploaded_files] + [file.id for file in baseline_uploaded_files]
                            code_interpreter = CodeInterpreterTool(file_ids=all_file_ids)

                                # Update the agent with the new code interpreter
                            agent_definition = await client.agents.update_agent(
                                agent_id=agent.id,
                                tools=code_interpreter.definitions,
                                tool_resources=code_interpreter.resources,
                            )

                            # Update the agent with the new definition
                            agent = AzureAIAgent(
                                client=client,
                                definition=agent_definition,
                            )
                        md_file.write("## Security Implementation Checklist\n\n")
                        md_file.write("For earch resource separately compare the security measures from your analysis and provide a table of security measures that are already present, missing, and need to be implemented:\n\n")
                        final_prompt = "For earch resource separately compare the security measures from your analysis and baseline files and provide a table of security measures that are already present, missing, and need to be implemented. The table should have the following columns: 'Security Measure', 'Present', 'Missing', 'Needs Implementation'. Provide response in markdown format and also include terraform attributes that need to be added for the recommendations.\n\n"
                        print(f"# User: '{final_prompt}'")
                            
                        async for response in agent.invoke(messages=final_prompt, thread=thread):
                            if response.role != AuthorRole.TOOL:
                                print(f"# Agent: {response}")
                                md_file.write(f"{response}\n\n")
                                thread = response.thread
    
            finally:
                # Cleanup: Delete the thread and agent
                await thread.delete() if thread else None
                await client.agents.delete_agent(agent.id)
            print("!!!!!!!!!!!!!!!!!HOGYAAAAAAAAAAAAAAAAAA!!!!!!!!!!!!!!!!!")



if __name__ == "__main__":
    asyncio.run(main())