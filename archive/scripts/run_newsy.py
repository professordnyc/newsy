#!/usr/bin/env python
"""
Newsy - Headlines Interrogation and Classification Agent

This standalone script orchestrates the entire news gathering and classification process
without relying on FastAPI/Uvicorn. It directly uses the service classes to:
1. Search for headlines using SerpAPI
2. Classify articles using Clarifai
3. Output the results to console and JSON file
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('newsy.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to Python path if not already there
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)
    logger.debug(f"Added {src_dir} to sys.path")

# Import services
try:
    from services.cache_service import CacheService
    from services.serpapi_service import SerpAPIService
    from services.article_service import ArticleService
    from services.clarifai_service import ClarifaiService, ClassificationType
    logger.debug("Successfully imported service modules")
except ImportError as e:
    logger.error(f"Error importing service modules: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

class NewsArticle:
    """Simple class to represent a news article with classification."""
    
    def __init__(self, title: str, url: str, source: str = None, snippet: str = None,
                 full_text: str = None, classification: str = None, 
                 confidence: float = 0.0, metadata: Dict = None):
        self.title = title
        self.url = url
        self.source = source
        self.snippet = snippet
        self.full_text = full_text
        self.classification = classification
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert the article to a dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "snippet": self.snippet,
            "full_text": self.full_text[:500] + "..." if self.full_text and len(self.full_text) > 500 else self.full_text,
            "classification": self.classification,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation of the article."""
        return f"{self.title} - {self.classification} ({self.confidence:.2f})"


class NewsyOrchestrator:
    """Main orchestrator for the Newsy workflow."""
    
    def __init__(self):
        """Initialize the orchestrator and its services."""
        logger.info("Initializing Newsy orchestrator...")
        
        # Initialize services
        self.cache_service = CacheService()
        self.serp_service = SerpAPIService(use_cache=True)
        self.article_service = ArticleService()
        self.clarifai_service = ClarifaiService()
        
        logger.info("All services initialized successfully")
    
    def search_and_classify(self, query: str, region: str = "us", 
                           max_results: int = 10, use_cache: bool = True) -> List[NewsArticle]:
        """
        Search for headlines and classify them.
        
        Args:
            query: Search query
            region: Region code (e.g., 'us', 'global')
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results
            
        Returns:
            List of classified NewsArticle objects
        """
        logger.info(f"Searching for '{query}' in region '{region}'...")
        
        # Search for headlines
        raw_results = self.serp_service.search_news(
            query=query,
            region=region,
            max_results=max_results,
            use_cache=use_cache
        )
        
        if not raw_results:
            logger.warning("No search results found")
            return []
            
        logger.info(f"Found {len(raw_results)} headlines")
        
        # Convert raw results to articles for classification
        articles_for_classification = []
        for idx, result in enumerate(raw_results):
            articles_for_classification.append({
                'id': str(idx),
                'title': result.get('title', ''),
                'text': result.get('snippet', ''),  # Using snippet as text for now
                'url': result.get('url', ''),
                'source': result.get('source', '')
            })
        
        # Batch classify the articles
        logger.info(f"Classifying {len(articles_for_classification)} articles...")
        classification_results = self.clarifai_service.batch_analyze_articles(
            articles_for_classification, 
            ClassificationType.LEADING_VS_UNDERREPORTED
        )
        
        # Create a lookup dictionary for classification results
        classifications = {}
        for result in classification_results:
            article_id = result.get('article_id')
            if article_id:
                classifications[article_id] = {
                    'prediction': result.get('prediction', 'unknown'),
                    'confidence': result.get('confidence', 0.0)
                }
        
        # Create NewsArticle objects with classifications
        classified_articles = []
        for idx, result in enumerate(raw_results):
            try:
                # Get classification for this article
                classification_data = classifications.get(str(idx), {})
                prediction = classification_data.get('prediction', 'unknown')
                confidence = classification_data.get('confidence', 0.0)
                
                # Create NewsArticle object
                article = NewsArticle(
                    title=result.get('title', ''),
                    url=result.get('url', ''),
                    source=result.get('source', ''),
                    snippet=result.get('snippet', ''),
                    classification=prediction,
                    confidence=confidence,
                    metadata={
                        'position': result.get('position'),
                        'raw_date': result.get('date'),
                        'thumbnail': result.get('thumbnail'),
                        'favicon': result.get('favicon')
                    }
                )
                
                classified_articles.append(article)
                
            except Exception as e:
                logger.warning(f"Error processing article {idx}: {str(e)}")
                continue
        
        logger.info(f"Successfully classified {len(classified_articles)} articles")
        return classified_articles
    
    def save_results(self, articles: List[NewsArticle], filename: str = None) -> str:
        """
        Save the classified articles to a JSON file.
        
        Args:
            articles: List of NewsArticle objects
            filename: Output filename (optional)
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"newsy_results_{timestamp}.json"
        
        # Convert articles to dictionaries
        articles_dict = [article.to_dict() for article in articles]
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def display_results(self, articles: List[NewsArticle]) -> None:
        """
        Display the classified articles in the console.
        
        Args:
            articles: List of NewsArticle objects
        """
        if not articles:
            print("No articles found.")
            return
        
        print(f"\n=== Found {len(articles)} classified articles ===\n")
        
        # Group articles by classification
        leading = [a for a in articles if a.classification.lower() == 'leading']
        under_reported = [a for a in articles if a.classification.lower() == 'under-reported']
        other = [a for a in articles if a.classification.lower() not in ['leading', 'under-reported']]
        
        # Print leading news
        if leading:
            print(f"\n== LEADING NEWS ({len(leading)}) ==")
            for i, article in enumerate(leading):
                print(f"{i+1}. {article.title}")
                print(f"   Source: {article.source}")
                print(f"   URL: {article.url}")
                print(f"   Confidence: {article.confidence:.2f}")
                print()
        
        # Print under-reported news
        if under_reported:
            print(f"\n== UNDER-REPORTED NEWS ({len(under_reported)}) ==")
            for i, article in enumerate(under_reported):
                print(f"{i+1}. {article.title}")
                print(f"   Source: {article.source}")
                print(f"   URL: {article.url}")
                print(f"   Confidence: {article.confidence:.2f}")
                print()
        
        # Print other news
        if other:
            print(f"\n== OTHER NEWS ({len(other)}) ==")
            for i, article in enumerate(other):
                print(f"{i+1}. {article.title}")
                print(f"   Source: {article.source}")
                print(f"   Classification: {article.classification}")
                print(f"   URL: {article.url}")
                print()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Newsy - Headlines Interrogation and Classification Agent")
    parser.add_argument("--query", default="technology", help="Search query")
    parser.add_argument("--region", default="us", choices=["us", "global"], help="Region code")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum number of results")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--output", help="Output filename (optional)")
    
    args = parser.parse_args()
    
    try:
        # Create orchestrator
        orchestrator = NewsyOrchestrator()
        
        # Search and classify
        articles = orchestrator.search_and_classify(
            query=args.query,
            region=args.region,
            max_results=args.max_results,
            use_cache=not args.no_cache
        )
        
        # Display results
        orchestrator.display_results(articles)
        
        # Save results
        output_file = orchestrator.save_results(articles, args.output)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error running Newsy: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
