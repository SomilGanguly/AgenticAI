"""
Azure Search Indexer - Manages document indexing in Azure AI Search
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from azure.search.documents import SearchClient
from azure.search.documents import IndexDocumentsBatch
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class AzureSearchIndexer:
    """Manages document indexing in Azure AI Search"""
    
    def __init__(self, config):
        self.config = config
        self.search_client: Optional[SearchClient] = None
    
    async def initialize(self):
        """Initialize Azure AI Search client"""
        try:
            search_config = self.config.get_azure_search_config()
            
            if not search_config["admin_key"]:
                raise ValueError("Azure Search admin key not found in configuration")
            
            self.search_client = SearchClient(
                endpoint=search_config["endpoint"],
                index_name=search_config["index_name"],
                credential=AzureKeyCredential(search_config["admin_key"])
            )
            
            logger.info("Azure Search indexer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Search indexer: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index documents in Azure AI Search with retry logic
        
        Args:
            documents: List of processed documents to index
        
        Returns:
            Dictionary with indexing results
        """
        try:
            if not self.search_client:
                raise RuntimeError("Search client not initialized")
            
            if not documents:
                logger.warning("No documents to index")
                return {"success": True, "indexed_count": 0, "errors": []}
            
            # Prepare documents for indexing
            search_documents = []
            for doc in documents:
                search_doc = self._prepare_document_for_search(doc)
                search_documents.append(search_doc)
            
            # # Create batch for uploading
            # batch = IndexDocumentsBatch()
            # for doc in search_documents:
            #     batch.add_upload_documents([doc])
            
            # Upload documents to index
            logger.info(f"Indexing {len(search_documents)} documents...")
            
            result = self.search_client.upload_documents(documents=search_documents)
            
            # Process results
            success_count = 0
            errors = []
            
            for res in result:
                if res.succeeded:
                    success_count += 1
                else:
                    errors.append({
                        "key": res.key,
                        "error": res.error_message,
                        "status_code": res.status_code
                    })
                    logger.error(f"Failed to index document {res.key}: {res.error_message}")
            
            logger.info(f"Successfully indexed {success_count}/{len(search_documents)} documents")
            
            return {
                "success": success_count > 0,
                "total_documents": len(search_documents),
                "indexed_count": success_count,
                "error_count": len(errors),
                "errors": errors
            }
            
        except HttpResponseError as e:
            logger.error(f"HTTP error during indexing: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise
    
    def _prepare_document_for_search(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare document for Azure AI Search indexing"""
        try:
            # Map document fields to search index schema
            search_doc = {
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "category": doc["category"],
                "wiki_path": doc["wiki_path"],
                "wiki_id": doc["wiki_id"],
                "source_type": doc["source_type"],
            }
            
            # Handle last_modified field
            last_modified = doc.get("last_modified")
            if isinstance(last_modified, datetime):
                # Ensure it's UTC and ends with Z
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=None)
                    last_modified_str = last_modified.isoformat() + "Z"
                else:
                    last_modified_str = last_modified.astimezone().isoformat()
                    if last_modified_str.endswith("+00:00"):
                        last_modified_str = last_modified_str[:-6] + "Z"
            else:
                last_modified_str = str(last_modified)
            search_doc["last_modified"] = last_modified_str
            
            # Add content vector if present
            if "content_vector" in doc and doc["content_vector"]:
                search_doc["content_vector"] = doc["content_vector"]
            
            # # Add metadata as additional fields
            # metadata = doc.get("metadata", {})
            # if "project_name" in doc:
            #     search_doc["project_name"] = doc["project_name"]
            
            # Validate required fields
            required_fields = ["id", "title", "content"]
            for field in required_fields:
                if not search_doc.get(field):
                    raise ValueError(f"Required field '{field}' is missing or empty")
            
            # Ensure content is not too long
            max_content_length = 50000  # Azure Search limit
            if len(search_doc["content"]) > max_content_length:
                search_doc["content"] = search_doc["content"][:max_content_length]
                logger.warning(f"Content truncated for document {search_doc['id']}")
            
            return search_doc
            
        except Exception as e:
            logger.error(f"Failed to prepare document for search: {str(e)}")
            raise
    
    async def delete_documents_by_source(self, project_name: str, wiki_id: str) -> Dict[str, Any]:
        """
        Delete documents from a specific source (project/wiki)
        
        Args:
            project_name: Project name to filter by
            wiki_id: Wiki ID to filter by
        
        Returns:
            Dictionary with deletion results
        """
        try:
            if not self.search_client:
                raise RuntimeError("Search client not initialized")
            
            # Search for existing documents from this source
            filter_expression = f"wiki_id eq '{wiki_id}'"
            if project_name:
                filter_expression += f" and project_name eq '{project_name}'"
            
            search_results = self.search_client.search(
                search_text="*",
                filter=filter_expression,
                select=["id"],
                top=1000  # Maximum batch size
            )
            
            # Collect document IDs to delete
            doc_ids = [result["id"] for result in search_results]
            
            if not doc_ids:
                logger.info(f"No documents found to delete for source: {project_name}/{wiki_id}")
                return {"success": True, "deleted_count": 0}
            
            # Delete documents
            documents_to_delete = [{"id": doc_id} for doc_id in doc_ids]
            result = self.search_client.delete_documents(documents=documents_to_delete)
            
            success_count = sum(1 for res in result if res.succeeded)
            
            logger.info(f"Deleted {success_count}/{len(doc_ids)} documents for source: {project_name}/{wiki_id}")
            
            return {
                "success": success_count > 0,
                "deleted_count": success_count,
                "total_found": len(doc_ids)
            }
            
        except Exception as e:
            logger.error(f"Error deleting documents by source: {str(e)}")
            raise
    
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        try:
            if not self.search_client:
                raise RuntimeError("Search client not initialized")
            
            # Get total document count
            search_results = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=0  # Don't return documents, just count
            )
            
            total_count = search_results.get_count()
            
            # Get count by category
            categories = {}
            category_results = self.search_client.search(
                search_text="*",
                facets=["category"],
                top=0
            )
            
            if hasattr(category_results, 'get_facets'):
                category_facets = category_results.get_facets().get("category", [])
                for facet in category_facets:
                    categories[facet["value"]] = facet["count"]
            
            return {
                "total_documents": total_count,
                "categories": categories,
                "timestamp": str(datetime.utcnow())
            }
            
        except Exception as e:
            logger.error(f"Error getting index statistics: {str(e)}")
            return {"error": str(e)}
