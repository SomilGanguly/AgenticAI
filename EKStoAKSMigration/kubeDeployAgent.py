import subprocess
from typing import Annotated

import semantic_kernel as sk
from kubernetes import config
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function

import utilities as util

AGENT_NAME = "KubernetesDeploymentAgent"
TARGET_MANIFEST_FOLDER = "./TargetManifests/"

class KubernetesDeploymentPlugin:
    def __init__(self):
        try:
            config.load_kube_config()
        except Exception as e:
            print(f"Error loading Kubernetes config: {e}")

    def run_kubectl_command(self, command: list) -> str:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running kubectl command: {e.stderr}")
            return "Failed"

    @kernel_function(description="Create specified Kubernetes resources in the given namespace and cluster.")
    def create_resource(
        self, resource_type: str, target_cluster: str, resource_name: str, namespace: str
    ) -> Annotated[str, "Returns details of the new Kubernetes resource."]:
        #Switch context to the specified cluster
        command = [ "kubectl", "config", "use-context", target_cluster]
        self.run_kubectl_command(command)

        # Check if namespace exists
        command = [ "kubectl", "get", "namespace", namespace.lower() ]
        result = self.run_kubectl_command(command)
        if result == "Failed":
            # Create namespace if it does not exist
            command = [ "kubectl", "create", "namespace", namespace.lower() ]
            result = self.run_kubectl_command(command)
            if result != "Failed":
                print(f"Namespace {namespace} created successfully.")
            else:
                return f"Failed to create namespace {namespace}."
        #Check if resource already exists
        command = [ "kubectl", "get", resource_type.lower(), "-n", namespace.lower() ]  # , "-o yaml >", resource_type + ".yaml"]
        result = self.run_kubectl_command(command)

        if result == "Failed":
            # Check if file exists
            output = util.read_file_if_exists(TARGET_MANIFEST_FOLDER + resource_type.lower() + "_" + resource_name.lower() + ".yaml")
            if output["metadata"]["name"] == resource_name:
                command = [ "kubectl", "create", "-f", TARGET_MANIFEST_FOLDER + resource_type.lower() + "_" + resource_name.lower() + ".yaml" ]
                result = self.run_kubectl_command(command)
                if result != "Failed":
                    return f"Resource {resource_type} with name {resource_name} created successfully in cluster {target_cluster}."
                # else:
                #     return f"Failed to create resource {resource_type} with name {resource_name} in cluster {target_cluster}."
            # else:
            #     return f"YAML file for resource {resource_type} with name {resource_name} does not exist in the target manifest folder."
        else:
            return f"Resource {resource_type} with name {resource_name} already exists in the cluster {target_cluster}."



def create_kubernetes_deployment_agent(
    instructions: str, deployment_name: str, endpoint: str, api_key: str
) -> ChatCompletionAgent:
    kernel = sk.Kernel()
    service = AzureChatCompletion(
        service_id="default",
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
    )
    kernel.add_service(service)
    kernel.add_plugin(KubernetesDeploymentPlugin(), plugin_name="KubernetesDeploymentPlugin")

    return ChatCompletionAgent(
        kernel=kernel, name=AGENT_NAME, instructions=instructions
    )
