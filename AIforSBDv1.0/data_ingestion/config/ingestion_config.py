"""
Configuration management for data ingestion process
"""
import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class IngestionConfig:
    """Configuration manager for data ingestion"""

    def __init__(self):
        self._config_file_path = Path(__file__).parent / "settings.json"
        self._config_data: Dict[str, Any] = {}

    async def initialize(self):
        """Initialize configuration from files"""
        try:
            await self._load_config_file()
            logger.info("Ingestion configuration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ingestion configuration: {str(e)}")
            raise

    async def _load_config_file(self):
        """Load configuration from JSON file"""
        try:
            if self._config_file_path.exists():
                with open(self._config_file_path, 'r') as f:
                    self._config_data = json.load(f)
                logger.info(f"Loaded configuration from {self._config_file_path}")
            else:
                await self._create_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration file: {str(e)}")
            raise

    async def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "azure_search": {
                "endpoint": "",
                "index_name": "",
                "semantic_config_name": "default",
                "admin_key": ""
            },
            "azure_devops": {
                "organization_url": "",
                "api_version": "7.1",
                "pat_token": ""
            },
            "processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_content_length": 50000,
                "supported_file_types": [".md", ".txt", ".html"]
            },
            "target_projects": [
                {
                    "name": "",
                    "type": "",
                    "wiki_id": "",
                    "path_filter": "*/**"
                }
            ],
            "logging": {
                "level": "INFO",
                "enable_telemetry": True
            }
        }

        with open(self._config_file_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        self._config_data = default_config
        logger.info(f"Created default configuration file at {self._config_file_path}")

    def get_azure_search_config(self) -> Dict[str, Any]:
        """Get Azure AI Search configuration"""
        config = self._config_data.get("azure_search", {})
        return {
            "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT") or config.get("endpoint"),
            "index_name": os.getenv("AZURE_SEARCH_INDEX") or config.get("index_name"),
            "admin_key": os.getenv("AZURE_SEARCH_ADMIN_KEY") or config.get("admin_key"),
            "semantic_config_name": config.get("semantic_config_name", "default")
        }

    def get_ado_config(self) -> Dict[str, Any]:
        """Get Azure DevOps configuration"""
        config = self._config_data.get("azure_devops", {})
        return {
            "organization_url": os.getenv("ADO_ORGANIZATION_URL") or config.get("organization_url"),
            "pat_token": os.getenv("ADO_PAT_TOKEN") or config.get("pat_token"),
            "api_version": config.get("api_version", "7.1")
        }

    def get_processing_config(self) -> Dict[str, Any]:
        """Get content processing configuration"""
        return self._config_data.get("processing", {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_content_length": 50000,
            "supported_file_types": [".md", ".txt", ".html"]
        })

    def get_target_projects(self) -> List[Dict[str, Any]]:
        """Get list of target projects for ingestion"""
        return self._config_data.get("target_projects", [])

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._config_data.get("logging", {
            "level": "INFO",
            "enable_telemetry": True
        })