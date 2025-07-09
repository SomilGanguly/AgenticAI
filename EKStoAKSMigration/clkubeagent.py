import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.functions import kernel_function

import utilities as util
import os

# Global reference to MCP plugin to keep it alive
kube_discovery_plugin = None
REPORT_DIR = "./ClusterReports/"

class ClusterReportPlugin:
    def __init__(self):
        pass

    @kernel_function(
        name="generate_cluster_summary",
        description="Generate a detailed Markdown summary of the Kubernetes cluster and save it to a file."
    )
    def generate_cluster_summary(self, summary_markdown: str, save_file: bool) -> str:
        if save_file:
            os.makedirs(REPORT_DIR, exist_ok=True)
            file_path = os.path.join(REPORT_DIR, "cluster_summary.md")
            util.write_to_file_md(file_path, summary_markdown)
            return f"Cluster summary document generated: `{file_path}`"
        else:
            return summary_markdown
        
async def create_kubernetes_discovery_agent(
    instructions: str, deployment_name: str, endpoint: str, api_key: str
) -> ChatCompletionAgent:
    """
    Create an agent that uses MCP to discover Kubernetes cluster resources.
    """

    global kube_discovery_plugin

    kernel = sk.Kernel()

    # Add Azure OpenAI completion service
    kernel.add_service(
        AzureChatCompletion(
            service_id="default",
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key=api_key,
        )
    )

    # Initialize MCP Kubernetes plugin
    kube_discovery_plugin = MCPStdioPlugin(
        name="kubernetes",
        description="Kubernetes discovery plugin",
        command="npx",
        args=["mcp-server-kubernetes"],
    )

    await kube_discovery_plugin.__aenter__()  # Ensure plugin is initialized

    # Register MCP plugin with the kernel
    kernel.add_plugin(kube_discovery_plugin, plugin_name="kubernetes_discovery_plugin")
    kernel.add_plugin(ClusterReportPlugin(), plugin_name="cluster_report_plugin")

    # Return configured agent
    return ChatCompletionAgent(
        kernel=kernel,
        name="KubernetesDiscoveryAgent",
        description="Discovers Kubernetes namespaces, pods, and other objects using MCP.",
        instructions=instructions,
    )
