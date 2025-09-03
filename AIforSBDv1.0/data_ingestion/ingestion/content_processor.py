"""
Content Processor - Processes and chunks wiki content for indexing
"""
import logging
import re
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import markdownify

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Processes wiki content for Azure AI Search indexing"""
    
    def __init__(self, config):
        self.config = config
        self.processing_config = config.get_processing_config()
    
    async def initialize(self):
        """Initialize content processor"""
        logger.info("Content processor initialized successfully")
    
    async def process_wiki_page(
        self, 
        page: Dict[str, Any], 
        project_name: str, 
        wiki_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Process a wiki page for indexing
        
        Args:
            page: Wiki page data from ADO
            project_name: Name of the project
            wiki_type: Type of wiki (standard, policies)
        
        Returns:
            Processed document ready for indexing
        """
        try:
            # Extract and clean content
            cleaned_content = await self._clean_content(page["content"])
            
            # Extract title from path or content
            title = await self._extract_title(page["path"], cleaned_content)
            
            # Determine category based on wiki type and content
            category = await self._determine_category(cleaned_content, wiki_type, page["path"])
            
            # Generate document ID
            doc_id = self._generate_document_id(page["path"], project_name, page["wiki_id"])
            
            # Create processed document
            processed_doc = {
                "id": doc_id,
                "title": title,
                "content": cleaned_content,
                "category": category,
                "wiki_path": page["path"],
                "wiki_id": page["wiki_id"],
                "source_type": wiki_type,
                "last_modified": datetime.utcnow(),
                "project_name": project_name,
                "content_length": len(cleaned_content),
                "metadata": {
                    "original_id": page.get("id"),
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "content_hash": hashlib.md5(cleaned_content.encode()).hexdigest()
                }
            }
            
            # Generate content vector (placeholder - would use embeddings in production)
            # processed_doc["content_vector"] = await self._generate_content_vector(cleaned_content)
            
            logger.info(f"Processed page: {page['path']} (category: {category})")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Failed to process page {page.get('path', 'unknown')}: {str(e)}")
            raise
    
    async def _clean_content(self, raw_content: str) -> str:
        """Clean and normalize wiki content"""
        try:
            # Remove HTML tags if present
            if "<" in raw_content and ">" in raw_content:
                soup = BeautifulSoup(raw_content, 'html.parser')
                content = soup.get_text()
            else:
                content = raw_content
            
            # Convert markdown to plain text if needed (preserve some formatting)
            if self._is_markdown(content):
                content = self._clean_markdown(content)
            
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n\s*\n', '\n\n', content)
            
            # Remove excessive line breaks
            content = re.sub(r'\n{3,}', '\n\n', content)
            
            # Trim and limit length
            content = content.strip()
            max_length = self.processing_config.get("max_content_length", 50000)
            if len(content) > max_length:
                content = content[:max_length] + "..."
                logger.warning(f"Content truncated to {max_length} characters")
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to clean content: {str(e)}")
            return raw_content[:1000]  # Return first 1000 chars as fallback
    
    def _is_markdown(self, content: str) -> bool:
        """Check if content is markdown format"""
        markdown_indicators = [
            r'#{1,6}\s',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\[.*?\]\(.*?\)',  # Links
            r'```',  # Code blocks
            r'^\s*[\*\-\+]\s',  # Lists
        ]
        
        for pattern in markdown_indicators:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False
    
    def _clean_markdown(self, content: str) -> str:
        """Clean markdown content while preserving structure"""
        # Convert to HTML first, then to clean text
        try:
            # Simple markdown cleaning - remove excessive formatting
            content = re.sub(r'#{1,6}\s*', '', content)  # Remove header markers
            content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold markers
            content = re.sub(r'\*(.*?)\*', r'\1', content)  # Remove italic markers
            content = re.sub(r'`(.*?)`', r'\1', content)  # Remove inline code markers
            content = re.sub(r'```.*?```', ' [CODE BLOCK] ', content, flags=re.DOTALL)  # Replace code blocks
            content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', content)  # Extract link text
            
            return content
        except Exception:
            return content
    
    async def _extract_title(self, path: str, content: str) -> str:
        """Extract title from path or content"""
        try:
            # Try to get title from path
            path_parts = path.strip('/').split('/')
            if path_parts and path_parts[-1]:
                title = path_parts[-1].replace('-', ' ').replace('_', ' ').title()
            else:
                title = "Wiki Page"
            
            # Try to get title from content (first line or heading)
            lines = content.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and len(line) < 100:  # Reasonable title length
                    # Remove markdown formatting
                    clean_line = re.sub(r'#{1,6}\s*', '', line)
                    clean_line = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_line)
                    if clean_line and not clean_line.startswith('['):
                        title = clean_line
                        break
            
            return title[:200]  # Limit title length
            
        except Exception as e:
            logger.warning(f"Failed to extract title: {str(e)}")
            return "Unknown Title"
    
    async def _determine_category(self, content: str, wiki_type: str, path: str) -> str:
        """Determine document category based on content and context"""
        try:
            # Default category based on wiki type
            if wiki_type == "policies":
                base_category = "policy"
            else:
                base_category = "documentation"
            
            # Analyze path for category hints
            path_lower = path.lower()
            path_categories = {
                "security": ["security", "auth", "authentication", "authorization"],
                "compliance": ["compliance", "audit", "governance", "standard"],
                "process": ["process", "procedure", "workflow", "guide"],
                "technical": ["technical", "api", "code", "development"],
                "policy": ["policy", "rule", "requirement", "regulation"]
            }
            
            for category, keywords in path_categories.items():
                if any(keyword in path_lower for keyword in keywords):
                    return category
            
            # Analyze content for category hints
            content_lower = content.lower()
            content_categories = {
                "security": ["password", "encryption", "vulnerability", "threat", "access control"],
                "compliance": ["must", "shall", "required", "mandatory", "compliance", "audit"],
                "process": ["step", "procedure", "process", "workflow", "guideline"],
                "technical": ["api", "code", "function", "class", "method", "technical"],
                "policy": ["policy", "rule", "standard", "requirement", "prohibited"]
            }
            
            for category, keywords in content_categories.items():
                keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
                if keyword_count >= 2:  # Require at least 2 keyword matches
                    return category
            
            return base_category
            
        except Exception as e:
            logger.warning(f"Failed to determine category: {str(e)}")
            return "general"
    
    def _generate_document_id(self, path: str, project_name: str, wiki_id: str) -> str:
        """Generate unique document ID"""
        # Create ID from path, project, and wiki ID
        id_string = f"{project_name}_{wiki_id}_{path}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    async def _generate_content_vector(self, content: str) -> List[float]:
        """Generate content vector using embeddings (placeholder)"""
        # In production, this would use Azure OpenAI embeddings
        # For now, return empty list - Azure AI Search can handle this
        return []
