import shutil
import subprocess

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.functions import kernel_function

from utilities import write_to_csv


class ACRPlugin:
    @kernel_function(
        name="import_image_to_acr",
        description="Import one or more Docker images into Azure Container Registry (ACR) from any OCI-compliant registry.",
    )
    def import_image_to_acr(self, acr_name: str, source_images: list[str]) -> str:
        # if len(source_images) != len(target_images):
        #     return "The number of source and target images must be equal."

        az_path = shutil.which("az")
        if not az_path:
            return "Azure CLI is not installed or not in PATH."

        results = []

        # for src, tgt in zip(source_images, target_images):
        for src in source_images:
            tgt = src.split("/")[-1]  # Use the image name as the target image name
            try:
                command = [
                    az_path,
                    "acr",
                    "import",
                    "--name",
                    acr_name,
                    "--source",
                    src,
                    "--image",
                    tgt,
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    results.append(f"Imported `{src}` as `{tgt}`")
                    write_to_csv(".\\Data\\ACRImages.csv", [[acr_name, tgt]])
                else:
                    results.append(f"Failed to import `{src}`: {result.stderr.strip()}")
            except Exception as e:
                results.append(f"Exception for `{src}`: {str(e)}")

        return "\n".join(results)


def create_acr_agent(instructions: str, deployment_name: str, endpoint: str, api_key: str):
    kernel = sk.Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="default",
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key=api_key,
        )
    )
    kernel.add_plugin(ACRPlugin(), plugin_name="acr")

    return ChatCompletionAgent(
        kernel=kernel,
        name="ACRAgent",
        description="Handles Docker image operations and Azure Container Registry tasks.",
        instructions=instructions
    )
