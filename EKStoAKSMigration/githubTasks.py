import asyncio
from datetime import date
import time

from azure.ai.projects.models import (Agent, AgentThread, AsyncToolSet,
                                      CodeInterpreterTool, FilePurpose,
                                      FileSearchTool)
from azure.identity import DefaultAzureCredential
from semantic_kernel.agents import (AzureAIAgent, AzureAIAgentSettings,
                                    AzureAIAgentThread)
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from utilities import load_instructions

# Convert the file to YAML and r

AGENT_NAME = "POCAgent"
INSTRUCTIONS_FILE = "manifest_instructions.txt"
USER_INPUTS = [
    "Create a new branch called testing in the using Github API if it does not already exist.",
    "How many deployments are there in the file?",
    # 'Check the "Manifests/Deployment.json" in the repository and replace the image in the container with the name "details" under the deployment "details-v1" with "NGINX". Store the updated file in the testing branch of the repository using Github API.',
    "Create a file called Deployment_Updated.json with the contents of the deployment called details_v1 and upload it to 'Updated Manifests' folder in the testing branch of the repository using Github API.",
    "Create a pull request from the branch testing to the main branch",
    "exit",
]
today = date.today()
# AGENT_INSTRUCTIONS = (
#     "Answer questions about the alishamb/agentic_AI github repository. The current date and time is: "
#     + str(today)
#     + "."
# )
toolset = AsyncToolSet()


async def add_agent_tools(project_client):
    file = await project_client.agents.upload_file_and_poll(
        file_path="./AKS Manifest/Deployment.json", purpose=FilePurpose.AGENTS
    )
    print(f"Uploaded file, file ID: {file.id}")

    vector_store = await project_client.agents.create_vector_store_and_poll(
        file_ids=[file.id], name="Deployment Vector Store"
    )
    print(f"Created vector store, vector store ID: {vector_store.id}")

    file_search = FileSearchTool(vector_store_ids=[vector_store.id])
    toolset.add(file_search)

    # code_interpreter = CodeInterpreterTool()
    # toolset.add(code_interpreter)
    return


async def intialize_agent(project_client, github_plugin) -> tuple[Agent, AgentThread]:
    instructions = load_instructions(INSTRUCTIONS_FILE)
    if instructions is None:
        print("Instructions file not found.")
        return None, None
    else:
        print("Instructions file loaded successfully.")
        print(instructions)
        await add_agent_tools(project_client)
        ai_agent_settings = AzureAIAgentSettings()
        print("Creating agent...")
        agent_definition = await project_client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name=AGENT_NAME,
            toolset=toolset,
            instructions=instructions,
            temperature=0.1,
        )
        # Create a Semantic Kernel agent based on the agent definition
        agent = AzureAIAgent(
            client=project_client, definition=agent_definition, plugins=[github_plugin]
        )
        print(f"Created agent, ID: {agent.id}")
        print("Creating thread...")
        thread: AzureAIAgentThread = AzureAIAgentThread(client=project_client)
        # thread = await project_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")
    return agent, thread


async def cleanup(agent: Agent, thread: AgentThread, project_client):
    """Cleanup the resources."""
    await thread.delete() if thread else None
    # await project_client.agents.delete_thread(thread.id)
    await project_client.agents.delete_agent(agent.id)
    return


async def main():
    creds = DefaultAzureCredential()
    async with AzureAIAgent.create_client(credential=creds) as client:
        async with MCPStdioPlugin(
            name="Github",
            description="Github Plugin",
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
            ],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": "PAT"
            },
        ) as github_plugin:
            agent, thread = await intialize_agent(client, github_plugin)
            if not agent or not thread:
                print("Agent or thread initialization failed.")

            # user_input = input("User:>")
            i = 0
            while (i>=0 and i<len(USER_INPUTS)) and USER_INPUTS[i] != "exit" and USER_INPUTS[i] != "save":
                user_input = USER_INPUTS[i]
                try:
                    print("i=", i)
                    print("User:>", user_input)
                    # while user_input not in ["exit", "save"] :
                    response = await agent.get_response(messages=user_input, thread=thread)
                    print(response)
                    thread = response.thread

                    # async for response in agent.invoke_stream(messages=user_input, thread=thread):
                    #     print(response, end = "")
                    # print("_____________________________________________________________________")
                    # message = await client.agents.create_message(
                    #     thread_id=thread.id,
                    #     role="user",
                    #     content=user_input
                    # )
                    # print(f"Created message, ID: {message.id}")
                    # run = await client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
                    # messages = await client.agents.list_messages(thread_id=thread.id)

                    # for text_message in messages.text_messages:
                    #     print(text_message.as_dict())

                    # user_input = input("User:>")
                    i += 1
                    # time.sleep(30)
                except Exception as e:
                    print(f"Error: {e}")
                    # time.sleep(20)
                    break
            user_input = USER_INPUTS[i]
            if user_input == "save":
                print(
                    "The agent has not been deleted, so you can continue experimenting with it in the Azure AI Foundry."
                )
            else:
                await cleanup(agent, thread, client)
                print("The agent resources have been cleaned up.")


if __name__ == "__main__":
    print("Starting async program...")
    asyncio.run(main())
    print("Program finished.")
