import os
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

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
    
    def __init__(self, pat: str = None, user_id: str = None, app_id: str = None):
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get values with the following priority: 1) passed arguments, 2) environment variables, 3) defaults
        self.pat = pat or os.getenv("CLARIFAI_PAT")
        self.user_id = user_id or os.getenv("CLARIFAI_USER_ID", "your_user_id_here")
        self.app_id = app_id or os.getenv("CLARIFAI_APP_ID", "newsy")
        
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
    
    def classify_text(self, text: str, model_id: str = "claude-3-sonnet") -> Optional[Dict]:
        if not text:
            logger.warning("Empty text provided for classification")
            return None
            
        if not self.pat:
            logger.error("CLARIFAI_PAT not set")
            return None
            
        model_map = {
            'claude-3-sonnet': 'claude-3-sonnet',
            'general-topics': 'general-topics',
            'sentiment-analysis': 'sentiment-analysis'
        }
        
        resolved_model_id = model_map.get(model_id, model_id)
        
        try:
            request = service_pb2.PostModelOutputsRequest(
                model_id=resolved_model_id,
                user_app_id=resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id),
                inputs=[resources_pb2.Input(
                    data=resources_pb2.Data(text=resources_pb2.Text(raw=text[:5000]))
                )]
            )
            
            response = self._make_request(request)
            if not response:
                return None
                
            return {
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'results': [json_format.MessageToDict(output) for output in response.outputs]
            }
            
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
        
    def batch_classify_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Classify multiple articles as leading or under-reported news.
        
        Args:
            articles: List of article dictionaries with 'title' and at least one of 'text', 'snippet', or 'full_text'
            
        Returns:
            List of dictionaries with article_id, url, and classification results
        """
        if not articles:
            logger.warning("No articles provided for batch classification")
            return []
            
        logger.info(f"Batch classifying {len(articles)} articles")
        results = []
        
        for idx, article in enumerate(articles):
            try:
                # Extract text content from the article
                title = article.get('title', '')
                
                # Try to get text content from various possible fields
                text = article.get('full_text', '')
                if not text:
                    text = article.get('text', '')
                if not text:
                    text = article.get('snippet', '')
                
                if not title and not text:
                    logger.warning(f"Article at index {idx} has no title or text content, skipping")
                    continue
                    
                # Prepare input for classification
                input_text = f"{title}\n\n{text}" if title and text else title or text
                
                # Classify the article
                classification = self.is_leading_news({'title': title, 'text': text})
                
                if classification:
                    # Extract the prediction and confidence
                    prediction = "unknown"
                    confidence = 0.0
                    
                    # Process the classification results
                    if 'results' in classification and classification['results']:
                        # Extract the prediction from the model output
                        # This depends on the specific format of your Clarifai model's output
                        # Adjust as needed based on your model's response structure
                        prediction = "leading"  # Default to leading if we can't determine
                        
                        # Try to extract from raw output
                        raw_output = classification['results'][0].get('output', {})
                        if 'data' in raw_output and 'text' in raw_output['data']:
                            text_output = raw_output['data']['text'].get('raw', '')
                            if 'under-reported' in text_output.lower():
                                prediction = "under_reported"
                            
                        # Try to get confidence
                        if 'data' in raw_output and 'concepts' in raw_output['data']:
                            concepts = raw_output['data']['concepts']
                            if concepts and len(concepts) > 0:
                                confidence = concepts[0].get('value', 0.0)
                    
                    results.append({
                        'article_id': article.get('id', str(idx)),
                        'url': article.get('url', ''),
                        'classification': {
                            'prediction': prediction,
                            'confidence': confidence
                        }
                    })
            except Exception as e:
                logger.error(f"Error classifying article at index {idx}: {str(e)}")
                
        logger.info(f"Successfully classified {len(results)} articles")
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
