import os
import logging
import hashlib
import json
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
from functools import lru_cache

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from google.protobuf import json_format
import grpc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationType(Enum):
    LEADING_VS_UNDERREPORTED = "leading_vs_underreported"
    TOPIC = "topic"
    SENTIMENT = "sentiment"

class ClarifaiService:
    """Service for interacting with Clarifai's AI models for news classification."""
    
    def __init__(self, pat: str = None, user_id: str = None, app_id: str = None, cache_size: int = 100, cache_ttl: int = 3600):
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get values with the following priority: 1) passed arguments, 2) environment variables, 3) defaults
        self.pat = pat or os.getenv("CLARIFAI_PAT")
        self.user_id = user_id or os.getenv("CLARIFAI_USER_ID", "your_user_id_here")
        self.app_id = app_id or os.getenv("CLARIFAI_APP_ID", "newsy")
        
        # Initialize cache settings
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl  # Time-to-live in seconds
        self.cache = {}  # Dictionary to store cached results
        self.cache_timestamps = {}  # Dictionary to store when each item was cached
        
        if not self.pat or self.pat == "your_personal_access_token_here":
            logger.error("CLARIFAI_PAT not properly configured. Please set it in your .env file")
            raise ValueError("CLARIFAI_PAT environment variable is required and must be set to your personal access token")
            
        try:
            self.channel = ClarifaiChannel.get_grpc_channel()
            self.stub = service_pb2_grpc.V2Stub(self.channel)
            self.metadata = (('authorization', f'Key {self.pat}'),)
            
            # Test connection
            test_request = service_pb2.ListModelsRequest(
                user_app_id=resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id),
                per_page=1
            )
            self.stub.ListModels(test_request, metadata=self.metadata, timeout=5)
            logger.info(f"Connected to Clarifai API (user_id: {self.user_id}, app_id: {self.app_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Clarifai service: {str(e)}")
            raise
            
    def _generate_cache_key(self, text: str, model_id: str) -> str:
        """Generate a unique cache key for the text and model combination."""
        # Create a hash of the text and model_id to use as a cache key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{model_id}_{text_hash}"
        
    def _get_from_cache(self, text: str, model_id: str) -> Optional[Dict]:
        """Try to get a result from the cache."""
        cache_key = self._generate_cache_key(text, model_id)
        
        # Check if the key exists in the cache
        if cache_key in self.cache:
            # Check if the cached item has expired
            timestamp = self.cache_timestamps.get(cache_key)
            if timestamp and (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                logger.info(f"Cache hit for model {model_id}")
                return self.cache[cache_key]
            else:
                # Remove expired item
                logger.debug(f"Removing expired cache item for model {model_id}")
                self.cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
                
        return None
        
    def _add_to_cache(self, text: str, model_id: str, result: Dict) -> None:
        """Add a result to the cache."""
        cache_key = self._generate_cache_key(text, model_id)
        
        # If cache is full, remove the oldest item
        if len(self.cache) >= self.cache_size:
            oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
            self.cache.pop(oldest_key, None)
            self.cache_timestamps.pop(oldest_key, None)
            
        # Add the new item to the cache
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.utcnow()
        logger.debug(f"Added result to cache for model {model_id}")
        
    def _clean_cache(self) -> None:
        """Remove expired items from the cache."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (now - timestamp).total_seconds() >= self.cache_ttl:
                expired_keys.append(key)
                
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
            
        logger.debug(f"Cleaned {len(expired_keys)} expired items from cache")
        
    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")
        
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        return {
            "size": len(self.cache),
            "max_size": self.cache_size,
            "ttl_seconds": self.cache_ttl
        }
    
    def _make_request(self, request, retries: int = 1, timeout: int = 10):
        attempt = 0
        while attempt <= retries:
            try:
                response = self.stub.PostModelOutputs(request, metadata=self.metadata, timeout=timeout)
                if response.status.code != 10000:  # SUCCESS code
                    logger.error(f"Clarifai API error: {response.status.description}")
                    return None
                return response
            except grpc.RpcError as rpc_err:
                logger.warning(f"Clarifai RPC error on attempt {attempt + 1}: {rpc_err}")
                attempt += 1
                if attempt > retries:
                    logger.error("Maximum retry attempts reached for Clarifai request")
                    return None
            except Exception as e:
                logger.error(f"Clarifai API request failed: {str(e)}")
                return None
    
    def classify_text(self, text: str, model_id: str = "llama-3", use_cache: bool = True, timeout: int = 15) -> Optional[Dict]:
        """Classify text using a Clarifai model with caching support.
        
        Args:
            text: The text to classify
            model_id: The ID of the model to use
            use_cache: Whether to use the cache
            timeout: Timeout in seconds for the API call
            
        Returns:
            Classification result dictionary or None if failed
        """
        if not text:
            logger.warning("Empty text provided for classification")
            return None
            
        if not self.pat:
            logger.error("CLARIFAI_PAT not set")
            return None
            
        # Check if we can use a cached result
        if use_cache:
            cached_result = self._get_from_cache(text, model_id)
            if cached_result:
                return cached_result
                
        # Periodically clean the cache
        if use_cache and len(self.cache) > self.cache_size / 2:
            self._clean_cache()
            
        # Map friendly names to Clarifai public model or workflow locations
        model_map = {
            # Llama-3 model
            'llama-3': {
                'type': 'model',
                'user_id': 'meta',
                'app_id': 'Llama-3',
                'id': 'Llama-3_2-3B-Instruct'
            },
            # General English Caption model - simpler and more reliable
            'general-english-caption': {
                'type': 'model',
                'user_id': 'clarifai',
                'app_id': 'main',
                'id': 'general-english-caption'
            },
            # Generic topics classifier (Clarifai community)
            'general-topics': {
                'type': 'model',
                'user_id': 'clarifai',
                'app_id': 'main',
                'id': 'general-topic-classification'
            },
            # Sentiment analysis
            'sentiment-analysis': {
                'type': 'model',
                'user_id': 'clarifai',
                'app_id': 'main',
                'id': 'sentiment'
            }
        }
        
        if model_id in model_map:
            mconf = model_map[model_id]
            req_user_id = mconf['user_id'] or self.user_id
            req_app_id = mconf['app_id'] or self.app_id
            resolved_id = mconf['id']
            item_type = mconf.get('type', 'model')
        else:
            req_user_id = self.user_id
            req_app_id = self.app_id
            resolved_id = model_id
            item_type = 'model'
        
        try:
            # Special handling for Llama-3
            if model_id == 'llama-3':
                # Create a chat-style input for Llama-3
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant that helps classify news articles. Analyze the following text and determine if it's a leading or under-reported news story. Respond with ONLY ONE WORD: either 'leading' or 'under-reported'.<|start_header_id|>user<|end_header_id|>\n\n{text[:2000]}<|end_of_text|>"
                
                request = service_pb2.PostModelOutputsRequest(
                    model_id=resolved_id,
                    user_app_id=resources_pb2.UserAppIDSet(user_id=req_user_id, app_id=req_app_id),
                    inputs=[resources_pb2.Input(
                        data=resources_pb2.Data(
                            text=resources_pb2.Text(raw=prompt)
                        )
                    )]
                )
                
                logger.debug(f"Sending request to Llama-3 with user_id={req_user_id}, app_id={req_app_id}, model_id={resolved_id}")
                try:
                    # Use the timeout parameter
                    response = self.stub.PostModelOutputs(request, metadata=self.metadata, timeout=timeout)
                    logger.debug(f"Received response type: {type(response)}")
                    
                    if not response:
                        logger.error("Empty response from Llama-3")
                        return None
                        
                    if not hasattr(response, 'outputs'):
                        logger.error(f"No outputs in Llama-3 response. Response: {response}")
                        return None
                        
                    outputs = response.outputs
                    logger.debug(f"Outputs count: {len(outputs)}")
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        logger.error(f"Timeout calling Llama-3 after {timeout} seconds")
                    else:
                        logger.error(f"gRPC error calling Llama-3: {e.code()}: {e.details()}")
                    return None
                except Exception as e:
                    logger.error(f"Error calling Llama-3: {str(e)}", exc_info=True)
                    return None
                
            elif item_type == 'workflow':
                request = service_pb2.PostWorkflowResultsRequest(
                    workflow_id=resolved_id,
                    user_app_id=resources_pb2.UserAppIDSet(user_id=req_user_id, app_id=req_app_id),
                    inputs=[resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw=text[:5000]))
                    )]
                )
            
            if item_type == 'workflow':
                try:
                    response = self.stub.PostWorkflowResults(request, metadata=self.metadata, timeout=10)
                    logger.debug(f"Raw workflow response type: {type(response)}")
                    logger.debug(f"Workflow response dir: {dir(response)}")
                    logger.debug(f"Workflow response: {response}")
                    
                    if not response:
                        logger.error("Empty workflow response")
                        return None
                        
                    if not hasattr(response, 'results') or not response.results:
                        logger.error(f"No results in workflow response. Response: {response}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Error processing workflow response: {str(e)}", exc_info=True)
                    return None
            else:
                response = self._make_request(request)
                if not response or not hasattr(response, 'outputs'):
                    logger.error(f"Invalid model response: {response}")
                    return None
                outputs = response.outputs
                
            # Process the response
            result = {
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'results': [json_format.MessageToDict(output) for output in outputs]
            }
            
            # For Llama-3, extract the text response and add classification
            if model_id == 'llama-3':
                try:
                    # Extract the text from the response
                    if result['results'] and 'data' in result['results'][0]:
                        text_response = result['results'][0]['data'].get('text', {}).get('raw', '')
                        logger.debug(f"Llama-3 text response: {text_response}")
                        
                        # Add a simple classification based on the response
                        if 'leading' in text_response.lower():
                            result['classification'] = 'leading'
                        elif 'under-reported' in text_response.lower():
                            result['classification'] = 'under-reported'
                        else:
                            result['classification'] = 'unknown'
                            
                        result['confidence'] = 0.9  # Placeholder confidence
                except Exception as e:
                    logger.error(f"Error processing Llama-3 response: {str(e)}", exc_info=True)
            
            # Add the result to the cache if caching is enabled
            if use_cache:
                self._add_to_cache(text, model_id, result)
                logger.debug(f"Added result to cache for model {model_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in classify_text: {str(e)}")
            return None
    
    def analyze_article(self, article: Dict, classification_type: ClassificationType) -> Dict:
        if not article or 'text' not in article:
            logger.warning("Invalid article format or missing text")
            return {}
            
        text = f"{article.get('title', '')}\n\n{article['text']}"
        
        model_map = {
            ClassificationType.LEADING_VS_UNDERREPORTED: "claude-3-sonnet",
            ClassificationType.TOPIC: "general-topics",
            ClassificationType.SENTIMENT: "sentiment-analysis"
        }
        
        model_id = model_map.get(classification_type, "claude-3-sonnet")
        result = self.classify_text(text, model_id)
        
        if not result or not result.get('results'):
            return {}
            
        processed_result = {
            'classification_type': classification_type.value,
            'model_used': model_id,
            'confidence': 0.0,
            'analysis': {}
        }
        # 2. Create a new Clarifai app and upload your labeled data
        # 3. Train a text classifier model using Clarifai's UI or API
        # 4. Replace the model_map above with your custom model ID
        #
        # Example implementation (commented out):
        #
        # model_map = {
        #     ClassificationType.LEADING_VS_UNDERREPORTED: "your-custom-model-id",
        #     ClassificationType.TOPIC: "your-topic-classifier-id",
        #     ClassificationType.SENTIMENT: "distilroberta-financial-news-sentiment-v2"
        # }
        #
        # # No need for prompt engineering with a fine-tuned model
        # # The model already knows how to classify the text
        
        return self.classify_text(text, model_id)
    
    def batch_analyze_articles(self, articles: List[Dict], classification_type: ClassificationType) -> List[Dict]:
        """
        Analyze multiple articles in a batch.
        
        Args:
            articles: List of article dictionaries
            classification_type: Type of analysis to perform
            
        Returns:
            List of analysis results
        """
        if not articles:
            return []
            
        results = []
        for article in articles:
            result = self.analyze_article(article, classification_type)
            if result:
                results.append({
                    'article_id': article.get('id'),
                    'url': article.get('url'),
                    'analysis': result
                })
                
        return results
        
    def _process_single_article(self, article_data: Tuple[int, Dict], model_id: str) -> Optional[Dict]:
        """Process a single article for classification (used in parallel processing).
        
        Args:
            article_data: Tuple of (index, article dictionary)
            model_id: ID of the model to use for classification
            
        Returns:
            Classified article dictionary or None if classification failed
        """
        idx, article = article_data
        try:
            # Extract text for classification
            title = article.get('title', '')
            summary = article.get('summary', '')
            content = article.get('content', '')
            
            # Prepare the text for classification
            text_to_classify = f"Title: {title}\n\nSummary: {summary}\n\nContent: {content}"
            
            # Classify the article using the specified model
            classification_result = self.classify_text(text_to_classify, model_id=model_id)
            if not classification_result:
                logger.warning(f"Failed to classify article: {title}")
                return None
                
            # Add classification to the article
            article_copy = article.copy()
            
            # Extract classification and confidence from the result
            classification = classification_result.get('classification', 'unknown')
            confidence = classification_result.get('confidence', 0.0)
            
            # Add to article
            article_copy['classification'] = classification
            article_copy['classification_confidence'] = confidence
            
            # Log the classification
            logger.info(f"Classified '{title}' as '{classification}' with confidence {confidence}")
            
            return article_copy
        except Exception as e:
            logger.error(f"Error classifying article at index {idx}: {str(e)}")
            return None
    
    def batch_classify_articles(self, articles: List[Dict], model_id: str = "llama-3", max_workers: int = 4) -> List[Dict]:
        """
        Classify multiple articles as leading or under-reported news using parallel processing.
        
        Args:
            articles: List of article dictionaries
            model_id: ID of the Clarifai model to use for classification
            max_workers: Maximum number of parallel workers for classification
            
        Returns:
            List of article dictionaries with added classification
        """
        if not articles:
            logger.warning("No articles provided for classification")
            return []
            
        logger.info(f"Batch classifying {len(articles)} articles using model {model_id} with {max_workers} workers")
        
        # Prepare article data with indices
        article_data = list(enumerate(articles))
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_article = {executor.submit(self._process_single_article, article, model_id): article 
                                for article in article_data}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_article):
                article_result = future.result()
                if article_result:
                    results.append(article_result)
        
        logger.info(f"Successfully classified {len(results)} articles using parallel processing")
        return results
    
    def is_leading_news(self, article: Dict) -> Dict:
        """
        Determine if an article is leading or underreported news.
        
        Args:
            article: Article data with 'title' and 'text' fields
            
        Returns:
            Dictionary with classification results
        """
        return self.analyze_article(article, ClassificationType.LEADING_VS_UNDERREPORTED)
        
    def batch_classify_leading_news(self, articles: List[Dict]) -> List[Dict]:
        """
        Classify multiple articles as leading or under-reported news.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of dictionaries with classification results
        """
        return self.batch_classify_articles(articles)
    
    def extract_topics(self, article: Dict) -> Dict:
        """Extract topics from an article."""
        return self.analyze_article(article, ClassificationType.TOPIC)
    
    def analyze_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment of an article."""
        return self.analyze_article(article, ClassificationType.SENTIMENT)
