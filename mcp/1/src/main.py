from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Newsy MCP Server",
    description="MCP server for Newsy Headlines Interrogation and Classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class HeadlineRequest(BaseModel):
    query: str
    region: str = "us"
    max_results: int = 10

class HeadlineResponse(BaseModel):
    title: str
    source: str
    date: str
    url: str
    summary: Optional[str] = None
    category: Optional[str] = None

# Routes
@app.get("/")
async def root():
    return {
        "service": "Newsy MCP Server",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/search/headlines", response_model=List[HeadlineResponse])
async def search_headlines(request: HeadlineRequest):
    """Search for news headlines based on query and region"""
    # TODO: Implement actual search logic with SerpAPI
    # This is a mock response for now
    return [
        {
            "title": "Sample Headline 1",
            "source": "Example News",
            "date": "2025-06-17",
            "url": "https://example.com/1",
            "summary": "This is a sample news article summary.",
            "category": "politics"
        },
        {
            "title": "Sample Headline 2",
            "source": "Sample Daily",
            "date": "2025-06-16",
            "url": "https://example.com/2",
            "summary": "Another example news summary.",
            "category": "technology"
        }
    ]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
