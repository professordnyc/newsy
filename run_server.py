"""Entry point to run the MCP server.
Starts Uvicorn programmatically to avoid import path issues that occur when running inside package directories.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "mcp.v1.src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
