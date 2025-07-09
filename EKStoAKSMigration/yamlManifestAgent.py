import subprocess
from typing import Annotated

import semantic_kernel as sk
from kubernetes import config
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function

import utilities as util

AGENT_NAME = "YamlManifestAgent"
SOURCE_MANIFEST_FOLDER = "./SourceManifests/"
TARGET_MANIFEST_FOLDER = "./TargetManifests/"
ACR_IMAGES_FILE = "./Data/ACRImages.csv"  # Path to the CSV file containing ACR images
ANNOTATIONS_FILE = "./Data/Annotations.csv"  # Path to the CSV file containing annotations


class YamlManifestPlugin:
    def __init__(self):
        try:
            config.load_kube_config()
        except Exception as e:
            print(f"Error loading Kubernetes config: {e}")

    def run_kubectl_command(self, command: list, save_file: bool) -> str:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if save_file:
                if command[3] != "-n" or command[3] != "--all-namespaces":
                    util.write_to_file(
                        SOURCE_MANIFEST_FOLDER
                        + command[2]
                        + "_"
                        + command[3]
                        + ".yaml",
                        result.stdout,
                    )
                else:
                    util.write_to_file(
                        SOURCE_MANIFEST_FOLDER + command[2] + ".yaml", result.stdout
                    )
                # with open(SOURCE_MANIFEST_FOLDER+command[2]+".yaml", "w") as file:
                #     file.write(result.stdout)
            else:
                print(f"Command: {' '.join(command)}")
                print(f"Output: {result.stdout}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running kubectl command: {e.stderr}")
            raise

    #  @kernel_function(description="Get details of Kubernetes resources.")
    def get_resource_details(
        self, resource_type: str, resource_name=None, namespace=None
    ) -> Annotated[str, "Returns details of the specified Kubernetes resource."]:
        command = [
            "kubectl",
            "get",
            resource_type,
        ]  # , "-o yaml >", resource_type + ".yaml"]
        if resource_name:
            command.append(resource_name)
        if namespace:
            command.extend(["-n", namespace])
        elif resource_type != "namespace":
            command.extend(["--all-namespaces"])
        command.append("-o")
        command.append("yaml")
        return self.run_kubectl_command(command, save_file=True)

    # @kernel_function(description="Creates Kubernetes resources using a manifest file.")
    # def create_resource(
    #     self, manifest_file: str
    # ) -> Annotated[str, "Returns output of the command."]:
    #     command = ["kubectl", "create", "-f", manifest_file]
    #     return self.run_kubectl_command(command)

    def change_deployment(
        self, resource_name: str, output, namespace: str = None
    ) -> Annotated[str, "Returns values to be updated for deployment."]:
        # Get Image from ACR
        values_to_update = []
        # Remove generic information from the deployment
        if "metadata" in output:
            if "creationTimestamp" in output["metadata"]:
                values_to_update.append({
                        "key": "creationTimestamp",
                        "old_value": output["metadata"]["creationTimestamp"],
                        "new_value": "na",
                        "action": "remove"
                    })
            if "status" in output:
                values_to_update.append({
                        "key": "status",
                        "old_value": output["status"],
                        "new_value": "na",
                        "action": "remove"
                    })
        # Consider multiple containers in the deployment
        container_count = len(
            output["spec"]["template"]["spec"]["containers"])
        for i in range(container_count):
            source_image = output["spec"]["template"]["spec"]["containers"][i][
                "image"
            ].split("/")[-1]
            acr_df = util.read_from_csv(ACR_IMAGES_FILE, {"Image": source_image})
            acr = acr_df.iloc[0]["ACR"] if not acr_df.empty else None
            if acr is None:
                values_to_update.append({
                    "key": "image",
                    "old_value": output["spec"]["template"]["spec"]["containers"][i]["image"],
                    "new_value": output["spec"]["template"]["spec"]["containers"][i]["image"],
                    "action": "replace"
                })
                if "imagePullSecrets" in output["spec"]["template"]["spec"]["containers"][i]:
                    values_to_update.append({
                        "key": "imagePullSecrets",
                        "old_value": output["spec"]["template"]["spec"]["containers"][i]["imagePullSecrets"],
                        "new_value": "na",
                        "action": "remove"
                    })
            else:
                values_to_update.append({
                    "key": "image",
                    "old_value": output["spec"]["template"]["spec"]["containers"][i]["image"],
                    "new_value": acr + ".azurecr.io/" + source_image,
                    "action": "replace"
                })

        # Get annotations from source manifest and delete any that contain 'aws'
        if "annotations" in output["metadata"]:
            annotations = output["metadata"]["annotations"]
            for key, value in annotations.items():
                if 'aws' in key.lower():
                    values_to_update.append({
                        "key": key,
                        "old_value": value,
                        "new_value": "na",
                        "action": "remove"
                    })
                if key == "kubectl.kubernetes.io/last-applied-configuration":
                    values_to_update.append({
                        "key": key.replace("/", "~1"),
                        "old_value": value,
                        "new_value": "na",
                        "action": "remove"
                    })
                # TODO: Add code to add new annotations
                # annotations_df = util.read_from_csv(ACR_IMAGES_FILE, {"EKS": key})
                # if not annotations_df.empty:
                #     values_to_update.append({
                #         "key": annotations_df.iloc[0]["AKS"],
                #         "old_value": "na",
                #         "new_value": annotations_df.iloc[0]["New_Value"],
                #         "action": "add"
                #     })
        print("values_to_update", values_to_update)
        return values_to_update

    @kernel_function(
        description="Identify changes needed in Kubernetes resources to migrate from EKS to AKS."
    )
    def identify_changes(
        self, resource_type: str, resource_name: str, acr_name: str, target_image: str, namespace: str = None,
    ) -> Annotated[str, "Returns changes in the specified Kubernetes resource."]:
        values_to_update =[]
        self.get_resource_details(resource_type, resource_name, namespace)
        output = util.read_file_if_exists(
            SOURCE_MANIFEST_FOLDER + resource_type + "_" + resource_name + ".yaml"
        )
        print("acr", acr_name)
        print("target_image", target_image)
        if output["metadata"]["name"] == resource_name:
            # TODO: Check if new image is available in ACR
            if resource_type == "deployment":
                values_to_update = self.change_deployment(
                    resource_name, output, namespace
                )

            if values_to_update != []:
                # Call function to create target file
                util.update_yaml_key(resource_type, output, values_to_update, TARGET_MANIFEST_FOLDER + resource_type + "_" + resource_name + ".yaml",
                )

                return f"Changes needed in {values_to_update} and target file created at {TARGET_MANIFEST_FOLDER}."
            else:
                return f"No changes identified for {resource_name} in {resource_type}."
        else:
            return "No resource_type resources to detect changes."


def create_yaml_manifest_agent(
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
    kernel.add_plugin(YamlManifestPlugin(), plugin_name="yamlManifestPlugin")

    return ChatCompletionAgent(
        kernel=kernel, name=AGENT_NAME, instructions=instructions
    )


# test = YamlManifestPlugin()
# test.identify_changes(
#     "deployment", "details-v1", "conmigcr", "examples-bookinfo-details-v1:1.16.4", "bookinfo"
# )
# test.identify_changes(
#     "deployment", "productpage-v1", "conmigcr", "examples-bookinfo-productpage-v1:1.16.4", "bookinfo"
# )
# test.identify_changes(
#     "deployment", "ratings-v1", "conmigcr", "examples-bookinfo-ratings-v1:1.16.4", "bookinfo"
# )

