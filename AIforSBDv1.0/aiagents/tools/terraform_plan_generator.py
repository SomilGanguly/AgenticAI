import os
import subprocess
import json
import tempfile
from typing import Dict, Any
import platform
import time
import re
import shutil

class TerraformPlanGenerator:
    """Generates Terraform plan files and converts them to JSON"""
    
    def __init__(self, terraform_path: str = None):
        # Try to find terraform executable
        self.terraform_path = terraform_path or self._find_terraform()
        if not self.terraform_path:
            raise ValueError("Terraform executable not found. Please install Terraform.")
    
    def _find_terraform(self) -> str:
        """Find terraform executable in PATH"""
        # Check if terraform is in PATH
        result = subprocess.run(['where' if platform.system() == 'Windows' else 'which', 'terraform'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]  # Take first result on Windows
        
        # Check common installation paths
        common_paths = [
            r"C:\terraform\terraform.exe",
            r"C:\Program Files\terraform\terraform.exe",
            r"C:\ProgramData\chocolatey\lib\terraform\tools\terraform.exe",
            "/usr/local/bin/terraform",
            "/usr/bin/terraform"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _extract_variables_from_tf_files(self, directory: str) -> Dict[str, Any]:
        """Extract variable definitions from .tf files to understand required variables"""
        variables = {}
        
        for file in os.listdir(directory):
            if file.endswith('.tf'):
                file_path = os.path.join(directory, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Improved regex to find variable blocks including multiline
                variable_pattern = r'variable\s+"([^"]+)"\s*\{([^}]*)\}'
                matches = re.finditer(variable_pattern, content, re.DOTALL)
                
                for match in matches:
                    var_name = match.group(1)
                    var_block = match.group(2)
                    
                    # Skip finding variables that already have defaults
                    has_default = 'default' in var_block
                    
                    if not has_default:
                        # Set common defaults based on variable name
                        if 'location' in var_name.lower():
                            variables[var_name] = 'eastus'
                        elif 'environment' in var_name.lower() or 'env' in var_name.lower():
                            variables[var_name] = 'dev'
                        elif 'project' in var_name.lower() or var_name.lower() == 'project_name':
                            variables[var_name] = 'test'
                        elif 'password' in var_name.lower():
                            variables[var_name] = 'P@ssw0rd123!'
                        else:
                            variables[var_name] = 'default'
        
        return variables
    
    def generate_plan(self, terraform_dir: str, tfvars_file: str = None) -> Dict[str, Any]:
        """
        Generate a Terraform plan and convert it to JSON
        
        Args:
            terraform_dir: Directory containing Terraform files
            tfvars_file: Optional path to .tfvars file
            
        Returns:
            Dict containing status and paths to generated files
        """
        original_dir = os.getcwd()
        temp_dir = None
        
        try:
            # Create a temporary copy of the terraform directory to avoid modifying the original
            temp_dir = tempfile.mkdtemp()
            print(f"Creating temporary workspace in: {temp_dir}")
            
            # Copy all terraform files to temp directory
            for root, dirs, files in os.walk(terraform_dir):
                for file in files:
                    if file.endswith(('.tf', '.tfvars')):
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, terraform_dir)
                        dst_path = os.path.join(temp_dir, rel_path)
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)
            
            # Change to temp directory
            os.chdir(temp_dir)
            
            # Fix duplicate provider configurations
            self._fix_duplicate_providers(temp_dir)
            
            # Remove or override backend configuration
            backend_file = os.path.join(temp_dir, "backend.tf")
            if os.path.exists(backend_file):
                print("Removing backend.tf to use local state...")
                os.remove(backend_file)
            
            # Also check for backend configuration in other files
            for tf_file in os.listdir(temp_dir):
                if tf_file.endswith('.tf'):
                    file_path = os.path.join(temp_dir, tf_file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Remove backend blocks
                    if 'backend "' in content:
                        print(f"Removing backend configuration from {tf_file}")
                        content = re.sub(r'backend\s+"[^"]+"\s*\{[^}]*\}', '', content, flags=re.DOTALL)
                        with open(file_path, 'w') as f:
                            f.write(content)
        
            # Initialize Terraform
            print("Initializing Terraform...")
            init_cmd = [self.terraform_path, 'init']
            init_result = subprocess.run(init_cmd, capture_output=True, text=True)
            
            if init_result.returncode != 0:
                return {
                    "status": "error",
                    "message": f"Terraform init failed: {init_result.stderr}"
                }
            
            # Extract variables that need values
            print("Analyzing required variables...")
            needed_variables = self._extract_variables_from_tf_files(temp_dir)
            
            # Generate plan
            print("Generating Terraform plan...")
            plan_file = "tf.plan"
            plan_cmd = [self.terraform_path, 'plan', '-out', plan_file, '-input=false']
            
            # If we have a tfvars file, use it first
            if tfvars_file and os.path.exists(tfvars_file):
                # Copy tfvars file to temp dir
                shutil.copy2(tfvars_file, temp_dir)
                plan_cmd.extend(['-var-file', os.path.basename(tfvars_file)])
                print(f"Using tfvars file: {os.path.basename(tfvars_file)}")
            
            # Add only the variables that are needed and not in tfvars
            for var_name, var_value in needed_variables.items():
                plan_cmd.extend(['-var', f'{var_name}={var_value}'])
                print(f"  Setting variable: {var_name}={var_value}")
            
            # First attempt
            plan_result = subprocess.run(plan_cmd, capture_output=True, text=True)
            
            if plan_result.returncode != 0:
                # Parse error to find missing variables
                missing_vars = re.findall(r'variable "([^"]+)" is not set', plan_result.stderr)
                undeclared_vars = re.findall(r'variable named "([^"]+)" was assigned', plan_result.stderr)
                
                # Remove undeclared variables from command
                if undeclared_vars:
                    print(f"Removing undeclared variables: {undeclared_vars}")
                    new_plan_cmd = []
                    i = 0
                    while i < len(plan_cmd):
                        if plan_cmd[i] == '-var' and i + 1 < len(plan_cmd):
                            var_assignment = plan_cmd[i + 1]
                            var_name = var_assignment.split('=')[0]
                            if var_name not in undeclared_vars:
                                new_plan_cmd.extend([plan_cmd[i], plan_cmd[i + 1]])
                            i += 2
                        else:
                            new_plan_cmd.append(plan_cmd[i])
                            i += 1
                    plan_cmd = new_plan_cmd
                
                # Add missing variables
                if missing_vars:
                    print(f"Adding missing variables: {missing_vars}")
                    for var in missing_vars:
                        if 'password' in var.lower():
                            plan_cmd.extend(['-var', f'{var}=P@ssw0rd123!'])
                        else:
                            plan_cmd.extend(['-var', f'{var}=default'])
                
                # Retry
                print("Retrying with adjusted variables...")
                plan_result = subprocess.run(plan_cmd, capture_output=True, text=True)
                
                if plan_result.returncode != 0:
                    # Try with refresh=false
                    print("Retrying with -refresh=false...")
                    plan_cmd.append('-refresh=false')
                    plan_result = subprocess.run(plan_cmd, capture_output=True, text=True)
                    
                    if plan_result.returncode != 0:
                        return {
                            "status": "error",
                            "message": f"Terraform plan failed: {plan_result.stderr}"
                        }
        
            # Convert plan to JSON
            print("Converting plan to JSON...")
            json_file = "terraform_file.json"
            show_result = subprocess.run([self.terraform_path, 'show', '-json', plan_file], 
                                       capture_output=True, text=True)
            
            if show_result.returncode != 0:
                return {
                    "status": "error",
                    "message": f"Terraform show failed: {show_result.stderr}"
                }
            
            # Save JSON output
            with open(json_file, 'w') as f:
                f.write(show_result.stdout)
            
            # Copy the JSON file to original directory
            final_json_path = os.path.join(terraform_dir, "terraform_file.json")
            shutil.copy2(json_file, final_json_path)
            
            return {
                "status": "success",
                "plan_file": os.path.join(temp_dir, plan_file),
                "json_file": final_json_path,
                "message": "Successfully generated Terraform plan and JSON"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to generate plan: {str(e)}"
            }
        finally:
            # Always restore original directory
            os.chdir(original_dir)
            
            # Clean up temp directory with retry logic
            if temp_dir and os.path.exists(temp_dir):
                for i in range(3):
                    try:
                        time.sleep(0.5)  # Give time for processes to release
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        break
                    except:
                        if i == 2:  # Last attempt
                            print(f"Warning: Could not clean up temp directory: {temp_dir}")

    def _fix_duplicate_providers(self, temp_dir: str):
        """Fix duplicate provider configurations by keeping only providers.tf"""
        providers_file = os.path.join(temp_dir, "providers.tf")
        main_file = os.path.join(temp_dir, "main.tf")
        
        if os.path.exists(providers_file) and os.path.exists(main_file):
            print("Fixing duplicate provider configurations...")
            
            # Read main.tf
            with open(main_file, 'r') as f:
                main_content = f.read()
            
            # Remove terraform block from main.tf
            main_content = re.sub(
                r'terraform\s*\{[^}]*required_providers\s*\{[^}]*\}[^}]*\}',
                '',
                main_content,
                flags=re.DOTALL
            )
            
            # Remove provider blocks from main.tf
            main_content = re.sub(
                r'provider\s+"[^"]+"\s*\{[^}]*\}',
                '',
                main_content,
                flags=re.DOTALL
            )
            
            # Write back cleaned main.tf
            with open(main_file, 'w') as f:
                f.write(main_content)
            
            print("Removed duplicate provider configurations from main.tf")