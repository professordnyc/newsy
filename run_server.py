"""Entry point to run the MCP server.
Starts Uvicorn programmatically to avoid import path issues that occur when running inside package directories.
"""
import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_server")

# Add the project root directory to Python path to fix import issues
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to Python path")

if __name__ == "__main__":
    logger.info("Starting Newsy MCP server...")
    try:
        uvicorn.run(
            "mcp.v1.src.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug",
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
