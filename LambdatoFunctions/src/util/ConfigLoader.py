import json
import os
from pathlib import Path
from src.util.AgentFactory import AgentConfig
from typing import List, Optional, Dict, Any
import logging

class ConfigLoader:
    """Loads and validates configuration files for agent setup."""
    
    def __init__(self, config_path: str):
        """Initialize ConfigLoader with path validation."""
        if not config_path:
            raise ValueError("Config path cannot be empty")
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config = self.load_config()
        
    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load and validate JSON configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            
            # Validate basic structure
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a JSON object")
            
            self._validate_config_structure(config)
            return config
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in configuration file {self.config_path}: {e}")
            return None
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return None
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> None:
        """Validate the basic structure of the configuration."""
        # Check for required top-level keys if any
        if 'agents' in config and not isinstance(config['agents'], list):
            raise ValueError("'agents' must be a list")
    
    def get_agent_configs(self) -> List[AgentConfig]:
        """Convert JSON agent configurations to AgentConfig objects with validation."""
        if self.config is None:
            return []
        
        try:
            agents_data = self.config.get('agents', [])
            if not agents_data:
                logging.warning("No agents found in configuration")
                return []
            
            agent_configs = []
            
            for i, agent_data in enumerate(agents_data):
                try:
                    # Validate required fields
                    if not self._validate_agent_data(agent_data, i):
                        continue
                    
                    # Handle instructions from file or direct
                    instructions = self._get_instructions(agent_data)
                    if not instructions:
                        logging.warning(f"No instructions found for agent {i}")
                        continue
                    
                    agent_config = AgentConfig(
                        name=agent_data['name'],
                        instructions=instructions,
                        model=agent_data['model'],
                        file_writer=agent_data.get('file_writer', False),
                        code_interpreter=agent_data.get('code_interpreter', False),
                        description=agent_data.get('description', "")
                    )
                    agent_configs.append(agent_config)
                    
                except Exception as e:
                    logging.error(f"Error processing agent {i}: {e}")
                    continue
                
            return agent_configs
            
        except Exception as e:
            logging.error(f"Error converting agent configurations: {e}")
            return []
    
    def _validate_agent_data(self, agent_data: Dict[str, Any], index: int) -> bool:
        """Validate individual agent data structure."""
        if not isinstance(agent_data, dict):
            logging.error(f"Agent {index} configuration must be an object")
            return False
        
        required_fields = ['name', 'model']
        for field in required_fields:
            if field not in agent_data:
                logging.error(f"Agent {index} missing required field: {field}")
                return False
            if not agent_data[field] or not isinstance(agent_data[field], str):
                logging.error(f"Agent {index} field '{field}' must be a non-empty string")
                return False
        
        # Validate that agent has either 'instructions' or 'instructions_file'
        if 'instructions' not in agent_data and 'instructions_file' not in agent_data:
            logging.error(f"Agent {index} must have either 'instructions' or 'instructions_file'")
            return False
        
        return True
    
    def _get_instructions(self, agent_data: Dict[str, Any]) -> str:
        """Get instructions either from direct field or file."""
        if 'instructions_file' in agent_data:
            return self._read_instruction_file(agent_data['instructions_file'])
        return agent_data.get('instructions', '')
    
    def get_agent_config_by_name(self, name: str) -> Optional[AgentConfig]:
        """Get a specific agent configuration by name with validation."""
        if not name:
            logging.warning("Agent name cannot be empty")
            return None
        
        agent_configs = self.get_agent_configs()
        for config in agent_configs:
            if config.name == name:
                return config
        
        logging.warning(f"Agent '{name}' not found in configuration")
        return None 
    
    def get_base_model(self) -> str:
        """Get the base model from the configuration with validation."""
        if self.config is None:
            return "gpt-4o"  # Default fallback
        
        model = self.config.get('base_model', 'gpt-4o')
        if not isinstance(model, str) or not model.strip():
            logging.warning("Invalid base_model in configuration, using default")
            return "gpt-4o"
        
        return model.strip()
    
    def get_task(self) -> str:
        """Get the task from the configuration with validation."""
        if self.config is None:
            return 'No task specified'
        
        task = self.config.get('initial_message', 'No task specified')
        if not isinstance(task, str):
            logging.warning("Invalid initial_message in configuration")
            return 'No task specified'
        
        return task

    def can_upload(self) -> bool:
        """Get the uploads setting from the configuration with validation."""
        if self.config is None:
            return False
        
        uploads = self.config.get('uploads', False)
        if not isinstance(uploads, bool):
            logging.warning("Invalid uploads setting in configuration, defaulting to False")
            return False
        
        return uploads
    
    def _read_instruction_file(self, file_name: str) -> str:
        """Read instruction file with enhanced error handling and validation."""
        if not file_name:
            logging.error("Instruction file name cannot be empty")
            return ""
        
        try:
            # Use pathlib for better path handling
            config_dir = self.config_path.parent
            file_path = config_dir / file_name
            
            # Validate file path security (prevent directory traversal)
            if not self._is_safe_path(file_path, config_dir):
                logging.error(f"Unsafe file path detected: {file_path}")
                return ""
            
            if not file_path.exists():
                logging.error(f"Instruction file not found: {file_path}")
                return ""
            
            if not file_path.is_file():
                logging.error(f"Instruction path is not a file: {file_path}")
                return ""
            
            # Check file size (prevent loading extremely large files)
            file_size = file_path.stat().st_size
            max_size = 10 * 1024 * 1024  # 10MB limit
            if file_size > max_size:
                logging.error(f"Instruction file too large ({file_size} bytes): {file_path}")
                return ""
            
            with open(file_path, 'r', encodiinstructions_fileng='utf-8') as file:
                instructions = file.read().strip()
            
            if not instructions:
                logging.warning(f"Instruction file is empty: {file_path}")
            
            return instructions
            
        except PermissionError:
            logging.error(f"Permission denied reading instruction file: {file_path}")
            return ""
        except UnicodeDecodeError:
            logging.error(f"Unable to decode instruction file (invalid UTF-8): {file_path}")
            return ""
        except Exception as e:
            logging.error(f"Error reading instruction file {file_path}: {e}")
            return ""
    
    def _is_safe_path(self, file_path: Path, base_dir: Path) -> bool:
        """Check if the file path is safe (within base directory)."""
        try:
            # Resolve both paths to handle symbolic links and relative paths
            resolved_file = file_path.resolve()
            resolved_base = base_dir.resolve()
            
            # Check if the file is within the base directory
            return resolved_file.is_relative_to(resolved_base)
        except (OSError, ValueError):
            return False
