import os
import re
import time
from pinecone import Pinecone, ServerlessSpec, PodSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
# This is where you add/remove candidates - no code changes needed elsewhere!
CANDIDATES_CONFIG = {
    "danilo": {
        "name": "Danilo Reitano",
        "cv_file": "Danilo_Reitano_CV.txt",
        "keywords": [r'\bdanilo\b', r'\breitano\b', r'\bdanilo reitano\b']
    },
    "rodrigo": {
        "name": "Rodrigo Mesa",
        "cv_file": "Rodrigo_Mesa_CV.txt",
        "keywords": [r'\brodrigo\b', r'\bmesa\b', r'\brodrigo mesa\b']
    },
    "juan": {
        "name": "Juan Garcia",
        "cv_file": "Juan_Garcia_CV.txt",
        "keywords": [r'\bjuan\b', r'\bgarcia\b', r'\bjuan garcia\b', r'\bignacio\b']
    }
    # To add a new candidate, just add a new entry here:
    # "maria": {
    #     "name": "Maria Lopez",
    #     "cv_file": "Maria_Lopez_CV.txt",
    #     "keywords": [r'\bmaria\b', r'\blopez\b']
    # }
}

def get_candidates_config():
    """Returns the candidates configuration."""
    return CANDIDATES_CONFIG

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
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1") 
            use_serverless = os.getenv("PINECONE_USE_SERVERLESS", "true").lower() == "true"
            
            if use_serverless:
                spec = ServerlessSpec(cloud=cloud, region=region)
            else:
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
                while not pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
            except Exception as e:
                print(f"Error creating index: {e}")
                raise e
        else:
            raise ValueError(f"Index '{index_name}' does not exist and create_if_not_exists is False.")
    
    return pc.Index(index_name)

def detect_candidate(query: str) -> str:
    """
    Detects which candidate the query is about using regex pattern matching.
    Dynamically uses CANDIDATES_CONFIG, so adding new candidates is automatic.
    
    Returns:
        str: candidate_id from config, "general", or "unknown"
    """
    query_lower = query.lower()
    
    # Check each configured candidate
    for candidate_id, config in CANDIDATES_CONFIG.items():
        for pattern in config["keywords"]:
            if re.search(pattern, query_lower):
                return candidate_id
    
    # Check for general/comparative queries
    general_keywords = [
        r'\bwho\b',
        r'\bquien\b',
        r'\bwhich candidate\b',
        r'\bcual\b',
        r'\bque candidato\b',
        r'\bcompare\b',
        r'\bcompara\b',
        r'\ball candidates\b',
        r'\btodos los candidatos\b',
        r'\btodos\b',
        r'\beveryone\b',
        r'\bambos\b',
        r'\bboth\b',
        r'\bcandidates\b',
        r'\bcandidatos\b',
        r'\bbest\b',
        r'\bworse\b',
        r'\bmejor\b',
        r'\bpeor\b',
    ]
    
    for keyword in general_keywords:
        if re.search(keyword, query_lower):
            return "general"
    
    # Default to general if no specific candidate detected
    return "general"
