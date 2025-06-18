from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class Region(str, Enum):
    US = "us"
    GLOBAL = "global"

class ClassificationType(str, Enum):
    LEADING = "leading"
    UNDER_REPORTED = "under_reported"
    UNKNOWN = "unknown"

class NewsArticle(BaseModel):
    """Represents a news article with its metadata and content."""
    title: str = Field(..., description="The headline/title of the article")
    url: HttpUrl = Field(..., description="URL to the full article")
    source: str = Field(..., description="Name of the news source")
    snippet: Optional[str] = Field(None, description="Brief summary/snippet from the article")
    publish_date: Optional[datetime] = Field(None, description="Publication date of the article")
    region: Optional[Region] = Field(None, description="Geographic region of the news")
    category: Optional[str] = Field(None, description="News category/topic")
    classification: Optional[ClassificationType] = Field(
        ClassificationType.UNKNOWN,
        description="Whether the article is leading or under-reported"
    )
    full_text: Optional[str] = Field(None, description="Full text content of the article")
    summary: Optional[str] = Field(None, description="AI-generated summary of the article")
    key_points: Optional[List[str]] = Field(None, description="Key points extracted from the article")
    sentiment: Optional[float] = Field(None, description="Sentiment score from -1 (negative) to 1 (positive)")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the article"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
        schema_extra = {
            "example": {
                "title": "Example News Article",
                "url": "https://example.com/news/article",
                "source": "Example News",
                "snippet": "This is an example news snippet...",
                "publish_date": "2023-10-25T12:00:00Z",
                "region": "us",
                "category": "politics",
                "classification": "leading",
                "full_text": "Full article text would appear here...",
                "summary": "This is an AI-generated summary of the article...",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "sentiment": 0.5,
                "metadata": {}
            }
        }
