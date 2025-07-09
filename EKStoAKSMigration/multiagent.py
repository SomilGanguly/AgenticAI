import os

import chainlit as cl
import semantic_kernel as sk
from azure.identity import DefaultAzureCredential
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatPromptExecutionSettings,
)

from acrAgent import create_acr_agent
from clkubeagent import create_kubernetes_discovery_agent
from utilities import get_kv_secret_value, load_instructions
from yamlManifestAgent import create_yaml_manifest_agent
from kubeDeployAgent import create_kubernetes_deployment_agent

YAML_AGENT_INSTRUCTIONS = "kubernetes_yaml_instructions.txt"
ORCHESTRATOR_INSTRUCTIONS = "orchestrator_instructions.txt"
ACR_INSTRUCTIONS = "acr_agent_instructions.txt"
KUBERNETES_INSTRUCTIONS = "kubernetes_discovery_instructions.txt"
DEPLOYMENT_AGENT_INSTRUCTIONS = "kubernetes_deployment_instructions.txt"


def get_settings():
    return OpenAIChatPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )


@cl.on_chat_start
async def on_chat_start():
    credential = DefaultAzureCredential()
    agent_settings = get_kv_secret_value(
        credential, ["AZURE-OPENAI-API-KEY", "AZURE-OPENAI-ENDPOINT"]
    )
    # Load orchestrator kernel
    orchestrator_kernel = sk.Kernel()
    orchestrator_kernel.add_service(
        AzureChatCompletion(
            service_id="default",
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            endpoint=agent_settings["AZURE-OPENAI-ENDPOINT"],
            api_key=agent_settings["AZURE-OPENAI-API-KEY"],
        )
    )

    acr_instructions = load_instructions(ACR_INSTRUCTIONS)
    acr_agent = create_acr_agent(
        instructions=acr_instructions,
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        endpoint=agent_settings["AZURE-OPENAI-ENDPOINT"],
        api_key=agent_settings["AZURE-OPENAI-API-KEY"],
    )

    instructions = load_instructions(YAML_AGENT_INSTRUCTIONS)
    yaml_agent = create_yaml_manifest_agent(
        instructions=instructions,
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        endpoint=agent_settings["AZURE-OPENAI-ENDPOINT"],
        api_key=agent_settings["AZURE-OPENAI-API-KEY"],
    )

    kubernetes_instructions = load_instructions(KUBERNETES_INSTRUCTIONS)
    kube_discovery_agent = await create_kubernetes_discovery_agent(
        instructions=kubernetes_instructions,
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        endpoint=agent_settings["AZURE-OPENAI-ENDPOINT"],
        api_key=agent_settings["AZURE-OPENAI-API-KEY"],
    )

    deployment_instructions = load_instructions(DEPLOYMENT_AGENT_INSTRUCTIONS)
    deployment_agent = create_kubernetes_deployment_agent(
        instructions=deployment_instructions,
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        endpoint=agent_settings["AZURE-OPENAI-ENDPOINT"],
        api_key=agent_settings["AZURE-OPENAI-API-KEY"],
    )

    # Orchestrator Agent
    orchestartor_instructions = load_instructions(ORCHESTRATOR_INSTRUCTIONS)
    orchestrator_agent = ChatCompletionAgent(
        instructions=orchestartor_instructions,
        kernel=orchestrator_kernel,
        name="Orchestrator",
        description="You are a orchestrator who can fetch all information about a kubernetes cluster and also appropriately use the agents to copy images, generate manifests etc based on user input..",
        plugins=[acr_agent, yaml_agent, kube_discovery_agent, deployment_agent],
    )

    cl.user_session.set("agent", orchestrator_agent)
    cl.user_session.set("thread", None)


# @cl.on_mcp_connect
# async def on_mcp_connect(connection, session: ClientSession):
#     """Handle MCP tool connection."""
#     result = await session.list_tools()
#     tools = [{
#         "name": t.name,
#         "description": t.description,
#         "input_schema": t.inputSchema,
#     } for t in result.tools]

#     mcp_tools = cl.user_session.get("mcp_tools", {})
#     mcp_tools[connection.name] = tools
#     cl.user_session.set("mcp_tools", mcp_tools)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    thread = cl.user_session.get("thread")

    answer = cl.Message(content="")

    async for response in agent.invoke_stream(messages=message.content, thread=thread):
        if response.content:
            await answer.stream_token(str(response.content))
        thread = response.thread
        cl.user_session.set("thread", thread)

    await answer.send()
