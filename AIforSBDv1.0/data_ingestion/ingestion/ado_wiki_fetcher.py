"""
ADO Wiki Fetcher - Retrieves content from Azure DevOps wikis
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from azure.devops.connection import Connection
from azure.devops.v7_1.wiki import WikiClient
from msrest.authentication import BasicAuthentication

logger = logging.getLogger(__name__)


class AdoWikiFetcher:
    """Fetches content from Azure DevOps wikis"""
    
    def __init__(self, config):
        self.config = config
        self.connection: Optional[Connection] = None
        self.wiki_client: Optional[WikiClient] = None
    
    async def initialize(self):
        """Initialize ADO connection and wiki client"""
        try:
            ado_config = self.config.get_ado_config()
            
            if not ado_config["pat_token"]:
                raise ValueError("ADO PAT token not found in configuration")
            
            # Create connection using PAT token
            credentials = BasicAuthentication('', ado_config["pat_token"])
            self.connection = Connection(
                base_url=ado_config["organization_url"],
                creds=credentials
            )
            
            # Get wiki client
            self.wiki_client = self.connection.clients.get_wiki_client()
            
            logger.info("ADO wiki fetcher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ADO wiki fetcher: {str(e)}")
            raise
    
    async def fetch_wiki_pages(
        self, 
        project_name: str, 
        wiki_id: str = None, 
        path_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch wiki pages from a specific project
        
        Args:
            project_name: Name of the ADO project
            wiki_id: Wiki ID (optional, will use default if not provided)
            path_filter: Path filter for pages (optional)
        
        Returns:
            List of wiki page dictionaries
        """
        try:
            if not self.wiki_client:
                raise RuntimeError("Wiki client not initialized")
            
            # Get project wikis
            wikis = self.wiki_client.get_all_wikis(project=project_name)
            
            if not wikis:
                logger.warning(f"No wikis found in project: {project_name}")
                return []
            
            logger.info(f"Wikis: {wikis[0]}")
            
            # Use specified wiki or first available
            target_wiki = None
            if wiki_id:
                target_wiki = next((w for w in wikis if w.id == wiki_id), None)
            else:
                target_wiki = wikis[0]  # Use first wiki
            
            if not target_wiki:
                logger.warning(f"Wiki not found in project: {project_name}")
                return []
            
            logger.info(f"Fetching pages from wiki: {target_wiki.name} in project: {project_name}")
            
            # Get all pages in the wiki
            pages = await self._get_wiki_pages_recursive(
                project_name=project_name,
                wiki_id=target_wiki.id,
                path="/",
                path_filter=path_filter
            )
            
            logger.info(f"Retrieved {len(pages)} pages from wiki: {target_wiki.name}")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to fetch wiki pages for project {project_name}: {str(e)}")
            return []
    
    async def _get_wiki_pages_recursive(
        self, 
        project_name: str, 
        wiki_id: str, 
        path: str,
        path_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Fetch all wiki pages by traversing the full page tree and retrieving content for each page."""
        pages = []
        try:
            # Get the full page tree (metadata only)
            page_tree_response = self.wiki_client.get_page(
                project=project_name,
                wiki_identifier=wiki_id,
                path=path,
                recursion_level=3,  # 2 = full tree
                include_content=False
            )
            # Helper function to traverse the tree recursively
            def traverse(page_node):
                if not page_node:
                    return
                page_path = page_node.path
                logger.info(f"Visiting page: {page_path}")
                # Always print subpage info
                subpages = getattr(page_node, "sub_pages", None)
                if subpages:
                    logger.info(f"Page {page_path} has {len(subpages)} sub_pages")
                else:
                    logger.info(f"Page {page_path} has no sub_pages")
                # Only fetch content for non-root pages
                if page_path and page_path != "/":
                    logger.info(f"Fetching content for page: {page_path}")
                    page_response = self.wiki_client.get_page(
                        project=project_name,
                        wiki_identifier=wiki_id,
                        path=page_path,
                        include_content=True
                    )
                    if page_response and page_response.page:
                        pages.append({
                            "id": page_response.page.id,
                            "path": page_path,
                            "content": page_response.page.content,
                            "last_modified": getattr(page_response.page, "git_item_path", None),
                            "project_name": project_name,
                            "wiki_id": wiki_id
                        })
                # Recursively traverse subpages
                if subpages:
                    for subpage in subpages:
                        traverse(subpage)
            # Start traversal from the root node
            if page_tree_response and page_tree_response.page:
                logger.info(f"Root page object: {page_tree_response.page}")
                logger.info(f"Root page dir: {dir(page_tree_response.page)}")
                logger.info(f"Root page dict: {page_tree_response.page.__dict__}")
                traverse(page_tree_response.page)
        except Exception as e:
            logger.error(f"Error fetching pages at path {path}: {str(e)}")
        if not pages:
            logger.info(f"No pages found at path {path} in wiki {wiki_id}")
        else:
            logger.info(f"Found {len(pages)} pages at path {path} in wiki {wiki_id}")
        return pages

    async def _get_page_content(self, project_name: str, wiki_id: str, path: str) -> Optional[str]:
        """Get content of a specific wiki page"""
        try:
            page_response = self.wiki_client.get_page(
                project=project_name,
                wiki_identifier=wiki_id,
                path=path,
                include_content=True
            )
            if page_response and page_response.page:
                logger.info(f"Retrieved content for page: {page_response.page.content}, path: {path}, wiki_id: {wiki_id}, project: {project_name}")
                return page_response.page.content
        except Exception as e:
            logger.error(f"Failed to get content for page {path}: {str(e)}")
            return None
    
    def _matches_path_filter(self, path: str, path_filter: str) -> bool:
        """Check if path matches the filter pattern"""
        if not path_filter:
            return True
        
        # Simple wildcard matching
        if path_filter.endswith("/**"):
            prefix = path_filter[:-3]
            return path.startswith(prefix)
        elif path_filter.endswith("/*"):
            prefix = path_filter[:-2]
            return path.startswith(prefix) and "/" not in path[len(prefix):].strip("/")
        else:
            return path == path_filter
