from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from pathlib import Path
import os

class FileWriter:
    def __init__(self ):
        self.__directory = os.path.join(Path(__file__).parents[2], "modified_files")

    @kernel_function(description="Create a file with the specified name and content in the directory.")
    def create_file(
        self, 
        file_name: Annotated[str, "The name of the file to create"],
        content: Annotated[str, "The content to write into the file", ""]
    ):
        """
        Create a file with the specified name and content in the directory.
        Args:
            file_name (str): The name of the file to create.
            content (str): The content to write into the file. Defaults to an empty string.
        Returns:
            str: The path to the created file.
        """
        directory = Path(self.__directory)
        file_path = directory / file_name
        
        # Create all parent directories for the file path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(file_path)