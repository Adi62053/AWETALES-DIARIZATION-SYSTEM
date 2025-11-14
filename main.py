import uvicorn
import logging
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Awetales Diarization System"""
    try:
        print("🚀 Starting Awetales Diarization System...")
        print("📁 Current directory:", os.getcwd())
        print(" Python path:", sys.executable)
        
        # Import and start the FastAPI app
        from src.api.app import app
        
        print(" FastAPI app imported successfully")
        print(" Server starting on http://localhost:8000")
        print(" API Docs: http://localhost:8000/docs")
        print("  Health: http://localhost:8000/api/v1/health")
        print("  Press Ctrl+C to stop the server")
        
        # Start the server with proper configuration
        uvicorn.run(
            "src.api.app:app",  # Use import string format
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f" Import error: {e}")
        print(" Try installing dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f" Failed to start server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
