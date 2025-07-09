import os
import asyncio
from dotenv import load_dotenv
from src.util.ConfigLoader import ConfigLoader
from src.util.AgentFactory import AgentFactory
from semantic_kernel.agents import AzureAIAgent, AgentGroupChat
from azure.identity.aio import DefaultAzureCredential
from src.util.gc_strat import ChatSelectionStrategy, ApprovalTerminationStrategy,generate_query_orchestrator_config 
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from azure.ai.projects.models import FilePurpose
from azure.ai.projects.models import CodeInterpreterTool
load_dotenv()
cwd=os.path.dirname(__file__)

async def main():
    print("Loading agent configurations...")
    folder_name = input("Enter the folder name for the configuration (default: 'input'): ") or 'input'
    config_path = os.path.join(cwd, folder_name, 'config.json')
    config_loader = ConfigLoader(config_path)
    agent_configs = config_loader.get_agent_configs()
    print("Configuration loaded successfully.")
    
    print("Client Initialization...")
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(
            credential=creds,
        ) as client,
    ):
        
        uploaded_files = []
        if config_loader.can_upload():
            files_path = os.path.join(cwd,folder_name, 'files')
            for root , dirs, files in os.walk(files_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    _file = await client.agents.upload_file_and_poll(file_path=file_path, purpose=FilePurpose.AGENTS)
                    uploaded_files.append(_file)
            print(f"Uploaded {len(uploaded_files)} files to the Azure AI Agent service.")
        
        code_interpreter = CodeInterpreterTool(file_ids=[file.id for file in uploaded_files])

        print("Client initialized successfully.")
        agent_factory = AgentFactory(client, code_interpreter)
        print("Building Agents...")
        agents = await agent_factory.create_agents(agent_configs)
        if len(agents) == 0:
            print("No agents were created. Please check your configuration.")
            return
        
        query_orch_config   = generate_query_orchestrator_config(agents, config_loader.get_base_model())
        query_orch_agent    = await agent_factory.create_agent(query_orch_config)
        
        try:
            chat = AgentGroupChat(
                agents = list(agents.values())+ [query_orch_agent],
                selection_strategy=ChatSelectionStrategy(
                    agents=list(agents.values()),
                    initial_agent=query_orch_agent,
                ),
                termination_strategy=ApprovalTerminationStrategy(
                    agents=[query_orch_agent],
                    maximum_iterations=10
                ),
            )
            print("Starting group chat...")
            await chat.add_chat_message(ChatMessageContent(
                content=config_loader.get_task(),
                role=AuthorRole.USER
            ))
            try:
                async for message in chat.invoke():
                    print(f"{message.name}: {message.content}")
            except Exception as e:
                print(f"Error during chat invocation: {e}")

        except Exception as e:
            print(f"Error during AgentGroupChat: {e}")
            return
        
        print("Agents created successfully.")
        
        #Group Chat Goes Here
        await agent_factory.cleanup()
        print("All agents cleaned up.")

if __name__ == "__main__":
    asyncio.run(main())