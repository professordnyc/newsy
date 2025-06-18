import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassificationType(str, Enum):
    """Types of classification tasks."""
    LEADING_VS_UNDERREPORTED = "leading_vs_underreported"
    TOPIC = "topic"
    SENTIMENT = "sentiment"

class ClarifaiService:
    """Service for interacting with Clarifai's AI models."""
    
    def __init__(self, pat: str = None, user_id: str = None, app_id: str = None):
        """
        Initialize the Clarifai service.
        
        Args:
            pat: Personal Access Token for Clarifai
            user_id: Clarifai user ID
            app_id: Clarifai app ID
        """
        self.pat = pat or os.getenv("CLARIFAI_PAT")
        self.user_id = user_id or os.getenv("CLARIFAI_USER_ID", "")
        self.app_id = app_id or os.getenv("CLARIFAI_APP_ID", "newsy")
        
        if not self.pat:
            logger.warning("CLARIFAI_PAT not found in environment variables")
            
        # Initialize the gRPC channel
        self.channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(self.channel)
        self.metadata = (('authorization', f'Key {self.pat}'),) if self.pat else ()
    
    def _make_request(self, request):
        """Send a request to the Clarifai API and handle the response."""
        try:
            response = self.stub.PostModelOutputs(
                request,
                metadata=self.metadata,
                timeout=30  # 30 seconds timeout
            )
            
            if response.status.code != status_code_pb2.SUCCESS:
                logger.error(f"Clarifai API error: {response.status.description}")
                return None
                
            return response
            
        except Exception as e:
            logger.error(f"Error calling Clarifai API: {str(e)}")
            return None
    
    def classify_text(self, text: str, model_id: str = "claude-3-sonnet") -> Optional[Dict]:
        """
        Classify text using a Clarifai model.
        
        Args:
            text: Text to classify
            model_id: ID of the Clarifai model to use
            
        Returns:
            Dictionary containing classification results or None if failed
        """
        if not text or not self.pat:
            return None
            
        request = service_pb2.PostModelOutputsRequest(
            model_id=model_id,
            user_app_id=resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id),
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=text[:5000])  # Limit text length
                    )
                )
            ]
        )
        
        response = self._make_request(request)
        if not response:
            return None
            
        # Convert response to dictionary
        result = {
            'model_id': model_id,
            'timestamp': datetime.utcnow().isoformat(),
            'results': []
        }
        
        for output in response.outputs:
            output_dict = json_format.MessageToDict(output)
            result['results'].append(output_dict)
            
        return result
    
    def analyze_article(self, article: Dict, classification_type: ClassificationType) -> Dict:
        """
        Analyze an article based on the specified classification type.
        
        Args:
            article: Article data with 'title' and 'text' fields
            classification_type: Type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        if not article or 'text' not in article:
            return {}
            
        text = f"{article.get('title', '')}\n\n{article['text']}"
        
        # Map classification types to appropriate models
        model_map = {
            ClassificationType.LEADING_VS_UNDERREPORTED: "news-classification",
            ClassificationType.TOPIC: "general-topics",
            ClassificationType.SENTIMENT: "sentiment-analysis"
        }
        
        model_id = model_map.get(classification_type, "claude-3-sonnet")
        
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
    
    def is_leading_news(self, article: Dict) -> Dict:
        """
        Determine if an article is leading or underreported news.
        
        Args:
            article: Article data
            
        Returns:
            Dictionary with classification results
        """
        return self.analyze_article(article, ClassificationType.LEADING_VS_UNDERREPORTED)
    
    def extract_topics(self, article: Dict) -> Dict:
        """Extract topics from an article."""
        return self.analyze_article(article, ClassificationType.TOPIC)
    
    def analyze_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment of an article."""
        return self.analyze_article(article, ClassificationType.SENTIMENT)
