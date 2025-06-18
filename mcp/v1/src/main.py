import os
import sys
import logging
from enum import Enum
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
import uvicorn

# Import models
from .models import NewsArticle, Region, ClassificationType

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('newsy_mcp.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent.parent.parent / "src")
if src_dir not in sys.path:
    sys.path.append(str(src_dir))
    
logger.debug(f"Added {src_dir} to sys.path")
logger.debug(f"Current sys.path: {sys.path}")

# Initialize services
def initialize_services():
    services = {}
    try:
        logger.info("Initializing services...")
        
        # Initialize cache first
        from services.cache_service import CacheService
        services['cache'] = CacheService()
        logger.info("Cache service initialized")
        
        # Initialize SerpAPI service
        from services.serpapi_service import SerpAPIService
        services['serp'] = SerpAPIService(use_cache=True)
        logger.info("SerpAPI service initialized")
        
        # Initialize Article service
        from services.article_service import ArticleService
        services['article'] = ArticleService()
        logger.info("Article service initialized")
        
        # Initialize Clarifai service
        from services.clarifai_service import ClarifaiService, ClassificationType
        services['clarifai'] = ClarifaiService()
        logger.info("Clarifai service initialized")
        
        return services, True
        
    except ImportError as e:
        logger.error(f"Import error initializing services: {e}", exc_info=True)
        return None, False
    except Exception as e:
        logger.error(f"Error initializing services: {e}", exc_info=True)
        return None, False

# Initialize all services
services, SERVICES_AVAILABLE = initialize_services()
if not SERVICES_AVAILABLE:
    logger.warning("Some services failed to initialize. Check logs for details.")

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Newsy MCP Server",
    description="MCP server for Newsy - A news classification and analysis service",
    version="0.1.0"
)

# Request models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for headlines")
    region: Region = Field(Region.US, description="Region for the search (us/global)")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results to return")
    use_cache: bool = Field(True, description="Whether to use cached results if available")
    
    # Add validator to handle case-insensitive region values
    @validator('region', pre=True)
    def normalize_region(cls, v):
        if isinstance(v, str):
            # Convert to lowercase and try to match
            v_lower = v.lower()
            if v_lower == "us":
                return Region.US
            elif v_lower == "global":
                return Region.GLOBAL
        return v

class ArticleExtractionRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of the article to extract")

class ArticleAnalysisRequest(BaseModel):
    title: str = Field(..., description="Title of the article")
    text: str = Field(..., description="Text content of the article")
    url: Optional[HttpUrl] = Field(None, description="URL of the article (if available)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to check service availability
async def check_services():
    if not SERVICES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required services are not available. Check server logs for details."
        )
    return services

# API endpoints
@app.get("/", include_in_schema=False)
def root():
    """Root endpoint with basic API information"""
    return {
        "name": "Newsy MCP Server",
        "version": "0.3.0",  # Changed from 0.1.0 to 0.3.0 to verify changes are applied
        "status": "running",  # Added status field
        "timestamp": datetime.utcnow().isoformat(),  # Added timestamp
        "services": {
            "serpapi": services.get('serp') is not None,
            "article_extraction": services.get('article') is not None,
            "clarifai": services.get('clarifai') is not None,
            "caching": services.get('cache') is not None
        }
    }

@app.get("/test/serpapi")
async def test_serpapi(services: dict = Depends(check_services)):
    """Test endpoint to check SerpAPI configuration and make a direct request."""
    try:
        # Get the SerpAPI service
        serp_service = services.get('serp')
        if not serp_service:
            return {"status": "error", "message": "SerpAPI service not available"}
            
        # Check API key
        api_key = serp_service.api_key
        if not api_key:
            return {"status": "error", "message": "SerpAPI key not found in environment variables"}
        
        # Log the first few characters of the API key for verification (never log the full key)
        masked_key = api_key[:4] + "*" * (len(api_key) - 4) if len(api_key) > 4 else "****"
        logger.debug(f"Using SerpAPI key: {masked_key}")
        
        # Make a direct request using the SerpAPI library
        from serpapi import GoogleSearch
        
        params = {
            "q": "test query",
            "api_key": api_key
        }
        
        logger.debug("Making direct test request to SerpAPI")
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Check for errors
        if "error" in results:
            return {"status": "error", "message": f"SerpAPI error: {results['error']}", "params": {k: v for k, v in params.items() if k != 'api_key'}}
        
        # Return success with available keys
        return {
            "status": "success", 
            "message": "SerpAPI test successful",
            "response_keys": list(results.keys() if isinstance(results, dict) else [])
        }
        
    except Exception as e:
        logger.error(f"Error testing SerpAPI: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.post("/search/headlines", response_model=List[NewsArticle])
async def search_headlines(
    request: SearchRequest,
    services: dict = Depends(check_services)
) -> List[NewsArticle]:
    """
    Search for news headlines using SerpAPI with optional caching.
    
    Args:
        request: Search parameters including query string, region, and result limits
        
    Returns:
        List of news articles matching the query with basic metadata
    """
    try:
        logger.info(f"=== Starting search for headlines ===")
        logger.info(f"Query: {request.query}")
        logger.info(f"Region: {request.region}")
        logger.info(f"Max results: {request.max_results}")
        logger.info(f"Use cache: {request.use_cache}")
        
        # Get the SerpAPI service
        serp_service = services.get('serp')
        if not serp_service:
            logger.error("SerpAPI service not available in services")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service is currently unavailable"
            )
            
        logger.info("Executing SerpAPI search...")
        
        # Perform the search
        results = serp_service.search_news(
            query=request.query,
            region=request.region,
            max_results=request.max_results,
            use_cache=request.use_cache
        )
        
        logger.info(f"Found {len(results)} results for query: {request.query}")
        if results:
            logger.debug(f"First result sample: {results[0]}")
        else:
            logger.warning("No results returned from SerpAPI")
            
        # Convert to NewsArticle objects
        articles = []
        for idx, result in enumerate(results):
            try:
                publish_date = None
                if 'date' in result and result['date']:
                    try:
                        publish_date = datetime.strptime(result['date'], '%Y-%m-%dT%H:%M:%S%z')
                    except (ValueError, TypeError):
                        publish_date = datetime.utcnow()

                # Handle source field which can be a dictionary or a string
                source = 'Unknown'
                if isinstance(result.get('source'), dict):
                    source = result.get('source', {}).get('name', 'Unknown')
                else:
                    source = str(result.get('source', 'Unknown'))
                
                article = NewsArticle(
                    title=result.get('title', '').strip() or "No title available",
                    url=result.get('url', result.get('link', '')),
                    source=source,
                    snippet=result.get('snippet', '').strip() or "No preview available",
                    publish_date=publish_date,
                    region=request.region,
                    category=result.get('category', 'general'),
                    metadata={
                        'search_position': idx + 1,
                        'search_query': request.query,
                        'source_url': result.get('link', ''),
                        'raw_date': result.get('date', '')  # Store the original date string
                    }
                )
                articles.append(article)
            except Exception as e:
                logger.error(f"Error processing article {idx}: {e}", exc_info=True)
                continue
        
        logger.info(f"Found {len(articles)} articles for query: {request.query}")
        return articles
        
    except Exception as e:
        logger.error(f"Error in search_headlines: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching for headlines: {str(e)}"
        )

@app.post("/articles/extract", response_model=NewsArticle)
async def extract_article(
    request: ArticleExtractionRequest,
    services: dict = Depends(check_services)
):
    """
    Extract full article content from a URL.
    
    Args:
        request: Contains the URL of the article to extract
        
    Returns:
        Extracted article with full content and metadata
    """
    try:
        url = str(request.url)
        logger.info(f"Extracting article from URL: {url}")
        
        # Extract article content
        extracted = services['article'].extract_article(url)
        
        if not extracted or not extracted.get('text'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Could not extract article content"
            )
        
        # Create a NewsArticle object with the extracted content
        article = NewsArticle(
            title=extracted.get('title', '').strip() or "No title available",
            url=url,
            source=extracted.get('source_url', '').strip() or "Unknown source",
            snippet=extracted.get('summary', extracted.get('text', '')[:200] + '...'),
            publish_date=extracted.get('publish_date'),
            full_text=extracted.get('text', ''),
            metadata={
                'authors': extracted.get('authors', []),
                'keywords': extracted.get('keywords', []),
                'images': extracted.get('images', []),
                'videos': extracted.get('videos', [])
            }
        )
        
        return article
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting article: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting article: {str(e)}"
        )

@app.post("/articles/analyze", response_model=NewsArticle)
async def analyze_article(
    request: ArticleAnalysisRequest,
    services: dict = Depends(check_services)
):
    """
    Analyze an article using Clarifai models.
    
    Args:
        request: Contains article title, text, and optional URL
        
    Returns:
        NewsArticle with analysis results including classification, summary, and key points
    """
    try:
        if not request.title and not request.text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either title or text must be provided"
            )
            
        logger.info(f"Analyzing article: {request.title[:50]}...")
        
        # Create base article
        article = NewsArticle(
            title=request.title,
            url=request.url,
            full_text=request.text,
            metadata={"analysis_started_at": datetime.utcnow().isoformat()}
        )
        
        # Get classification (leading vs under-reported)
        classification = services['clarifai'].classify_article(
            title=request.title,
            text=request.text,
            classification_type=ClassificationType.LEADING
        )
        if classification:
            article.classification = classification.get('prediction', ClassificationType.UNKNOWN)
            article.metadata['classification_confidence'] = classification.get('confidence', 0)
        
        # Get summary
        summary = services['clarifai'].summarize_text(
            text=request.text,
            title=request.title
        )
        if summary:
            article.summary = summary.get('summary', '')
            article.key_points = summary.get('key_points', [])
        
        # Get sentiment analysis
        sentiment = services['clarifai'].analyze_sentiment(
            title=request.title,
            text=request.text
        )
        if sentiment:
            article.sentiment = sentiment.get('score', 0)
            article.metadata['sentiment_label'] = sentiment.get('label', 'neutral')
        
        article.metadata['analysis_completed_at'] = datetime.utcnow().isoformat()
        
        return article
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing article: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing article: {str(e)}"
        )

# Simple test endpoint
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify route registration."""
    return {"status": "success", "message": "Test endpoint is working"}

# Health check endpoint
@app.get("/health")
async def health_check(services: dict = Depends(check_services)):
    """
    Health check endpoint that verifies all services are operational.
    
    Returns:
        Status of all services
    """
    status_checks = {
        "serpapi": services.get('serp') is not None,
        "article_extraction": services.get('article') is not None,
        "clarifai": services.get('clarifai') is not None,
        "caching": services.get('cache') is not None,
        "status": "healthy" if SERVICES_AVAILABLE else "degraded",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not SERVICES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "degraded", "services": status_checks}
        )
        
    return status_checks

# Clear cache endpoint (protected in production)
@app.post("/cache/clear")
async def clear_cache(
    services: dict = Depends(check_services),
    api_key: str = None
):
    """
    Clear the cache (protected endpoint).
    
    Args:
        api_key: API key for authentication (if required)
        
    Returns:
        Status of cache clearing operation
    """
    # In production, you would verify the API key here
    # if not is_valid_api_key(api_key):
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        if services.get('serp'):
            services['serp'].clear_cache()
        if services.get('cache'):
            services['cache'].clear()
            
        return {"status": "success", "message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "mcp.v1.src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )
