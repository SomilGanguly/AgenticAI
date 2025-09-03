import os
import tempfile
import shutil
from typing import Dict, Any
import git
from urllib.parse import urlparse
import stat
import time

class GitHubFetcher:
    """Fetches code from GitHub repositories"""
    
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.repo_path = None
    
    def fetch_repo(self, github_url: str) -> Dict[str, Any]:
        """
        Fetches a GitHub repository and returns the path to the cloned repo
        
        Args:
            github_url: The GitHub repository URL
            
        Returns:
            Dict containing status and local path
        """
        try:
            # Parse the URL to get repo name
            parsed_url = urlparse(github_url)
            repo_name = parsed_url.path.strip('/').split('/')[-1].replace('.git', '')
            
            # Create a directory for this repo
            self.repo_path = os.path.join(self.temp_dir, repo_name)
            
            # Remove if exists
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path, ignore_errors=True)
            
            # Clone the repository
            print(f"Cloning repository from {github_url}...")
            git.Repo.clone_from(github_url, self.repo_path)
            
            # Find Terraform files
            tf_files = []
            for root, dirs, files in os.walk(self.repo_path):
                # Skip .git and .terraform directories
                dirs[:] = [d for d in dirs if d not in ['.git', '.terraform']]
                for file in files:
                    if file.endswith('.tf') or file.endswith('.tfvars'):
                        tf_files.append(os.path.join(root, file))
            
            return {
                "status": "success",
                "repo_path": self.repo_path,
                "terraform_files": tf_files,
                "message": f"Successfully cloned repository to {self.repo_path}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to fetch repository: {str(e)}"
            }
    
    def cleanup(self):
        """Clean up temporary directory with retry logic"""
        if not os.path.exists(self.temp_dir):
            return
            
        def _on_rm_error(func, path, exc_info):
            """Error handler for Windows file removal"""
            try:
                os.chmod(path, stat.S_IWRITE)
                os.unlink(path)
            except:
                pass
        
        # Try multiple times with delays
        for attempt in range(3):
            try:
                # First, try to close any open handles
                import gc
                gc.collect()
                time.sleep(0.5)
                
                # Try to remove
                shutil.rmtree(self.temp_dir, onerror=_on_rm_error)
                print("Successfully cleaned up temporary files")
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"Warning: Could not fully clean up {self.temp_dir}: {e}")
                else:
                    time.sleep(1)  # Wait before retry