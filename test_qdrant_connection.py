from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    # Get Qdrant credentials from env
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"Connecting to Qdrant at: {qdrant_url}")
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Test connection by listing collections
        collections = client.get_collections()
        print("Connection successful!")
        print(f"Collections: {collections}")
        return True
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection() 