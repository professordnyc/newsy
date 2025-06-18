import logging
from typing import Dict, Optional, List
from datetime import datetime
from newspaper import Article, ArticleException
from urllib.parse import urlparse
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleService:
    """Service for extracting and processing article content using newspaper3k."""
    
    def __init__(self, language: str = 'en', fetch_images: bool = False, request_timeout: int = 10):
        """
        Initialize the ArticleService.
        
        Args:
            language: Language code for article processing
            fetch_images: Whether to fetch images (default: False to save bandwidth)
            request_timeout: Timeout in seconds for HTTP requests
        """
        self.language = language
        self.fetch_images = fetch_images
        self.request_timeout = request_timeout
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    def extract_article(self, url: str) -> Optional[Dict]:
        """
        Extract article content from a given URL.
        
        Args:
            url: URL of the article to extract
            
        Returns:
            Dictionary containing article data or None if extraction fails
        """
        if not url or not url.strip():
            return None
            
        article = Article(
            url=url,
            language=self.language,
            fetch_images=self.fetch_images,
            request_timeout=self.request_timeout
        )
        
        try:
            # Set a browser-like user agent
            article.set_user_agent(self.user_agent)
            
            # Download and parse the article
            article.download()
            article.parse()
            
            # Perform natural language processing (if needed)
            article.nlp()
            
            return {
                'url': url,
                'title': article.title,
                'text': article.text,
                'summary': article.summary,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'publish_date_raw': str(article.publish_date),
                'keywords': article.keywords,
                'source_url': article.source_url,
                'top_image': article.top_image,
                'images': article.images,
                'movies': article.movies,
                'meta_keywords': article.meta_keywords,
                'meta_description': article.meta_description,
                'canonical_link': article.canonical_link,
                'extraction_time': datetime.utcnow().isoformat(),
                'domain': urlparse(url).netloc if url else None
            }
            
        except ArticleException as e:
            logger.error(f"Error extracting article from {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {str(e)}")
            return None
    
    def extract_multiple_articles(self, urls: List[str], max_workers: int = 5) -> List[Dict]:
        """
        Extract multiple articles concurrently.
        
        Args:
            urls: List of article URLs to extract
            max_workers: Maximum number of concurrent extractions
            
        Returns:
            List of extracted article data (only successful extractions)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(self.extract_article, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    
        return results
    
    def is_valid_article(self, article_data: Dict, min_length: int = 100) -> bool:
        """
        Check if the extracted article meets basic quality criteria.
        
        Args:
            article_data: Article data from extract_article
            min_length: Minimum text length to be considered valid
            
        Returns:
            bool: True if the article meets the criteria
        """
        if not article_data:
            return False
            
        text = article_data.get('text', '').strip()
        title = article_data.get('title', '').strip()
        
        # Basic validation
        if len(text) < min_length:
            return False
            
        if not title:
            return False
            
        return True
