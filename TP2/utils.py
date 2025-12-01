import os
import time
from pinecone import Pinecone, ServerlessSpec, PodSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer model.
    """
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")
    return model

def get_pinecone_client() -> Pinecone:
    """
    Initializes and returns the Pinecone client.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
    
    return Pinecone(api_key=api_key)

def get_pinecone_index(index_name: str, dimension: int = 384, create_if_not_exists: bool = False):
    """
    Connects to a Pinecone index. Optionally creates it if it doesn't exist.
    """
    pc = get_pinecone_client()
    
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        if create_if_not_exists:
            print(f"Index '{index_name}' not found. Creating it...")
            # We'll default to serverless spec for simplicity, or allow config
            # Note: Serverless spec requires cloud and region
            # For this exercise, we'll try to use ServerlessSpec if available, or PodSpec
            # Assuming Serverless is the default preference for new users
            
            # Use environment variables for spec if needed, or default
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1") 
            
            # Check if we should use serverless or pod based on env
            use_serverless = os.getenv("PINECONE_USE_SERVERLESS", "true").lower() == "true"
            
            if use_serverless:
                spec = ServerlessSpec(cloud=cloud, region=region)
            else:
                # Fallback to pod spec (e.g. for starter environment)
                # But starter envs are now serverless on AWS us-east-1 usually
                # If using old pod-based starter:
                env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
                spec = PodSpec(environment=env, pod_type="starter")

            try:
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=spec
                )
                print(f"Index '{index_name}' created.")
                # Wait for index to be ready
                while not pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
            except Exception as e:
                print(f"Error creating index: {e}")
                # Fallback or re-raise
                raise e
        else:
            raise ValueError(f"Index '{index_name}' does not exist and create_if_not_exists is False.")
    
    return pc.Index(index_name)
