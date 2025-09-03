"""
Main data ingestion script for ADO wiki content
This runs independently from the Function App
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from ingestion.ado_wiki_fetcher import AdoWikiFetcher
from ingestion.content_processor import ContentProcessor
from indexing.azure_search_indexer import AzureSearchIndexer
from config.ingestion_config import IngestionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main ingestion process"""
    try:
        logger.info("Starting ADO wiki data ingestion process")
        
        # Load configuration
        config = IngestionConfig()
        await config.initialize()
        
        # Initialize components
        wiki_fetcher = AdoWikiFetcher(config)
        content_processor = ContentProcessor(config)
        search_indexer = AzureSearchIndexer(config)
        
        # Initialize services
        await wiki_fetcher.initialize()
        await content_processor.initialize()
        await search_indexer.initialize()
        
        # Get list of projects and wikis to process
        projects = config.get_target_projects()
        
        for project_config in projects:
            project_name = project_config["name"]
            wiki_type = project_config.get("type", "standard")  # standard or policies
            
            logger.info(f"Processing project: {project_name} (type: {wiki_type})")
            
            try:
                # Fetch wiki pages from ADO
                wiki_pages = await wiki_fetcher.fetch_wiki_pages(
                    project_name=project_name,
                    wiki_id=project_config.get("wiki_id"),
                    path_filter=project_config.get("path_filter")
                )
                
                logger.info(f"Fetched {len(wiki_pages)} pages from {project_name}")
                
                # Process content for each page
                processed_documents = []
                for page in wiki_pages:
                    try:
                        processed_doc = await content_processor.process_wiki_page(
                            page=page,
                            project_name=project_name,
                            wiki_type=wiki_type
                        )
                        processed_documents.append(processed_doc)
                        
                    except Exception as e:
                        logger.error(f"Failed to process page {page.get('path', 'unknown')}: {str(e)}")
                        continue
                
                logger.info(f"Processed {len(processed_documents)} documents from {project_name}")
                
                # Index documents in Azure AI Search
                if processed_documents:
                    await search_indexer.index_documents(processed_documents)
                    logger.info(f"Indexed {len(processed_documents)} documents from {project_name}")
                
            except Exception as e:
                logger.error(f"Failed to process project {project_name}: {str(e)}")
                continue
        
        logger.info("Data ingestion process completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in data ingestion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
