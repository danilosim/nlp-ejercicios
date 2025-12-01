import os
import glob
from typing import List, Dict
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import get_embedding_model, get_pinecone_index
from dotenv import load_dotenv

load_dotenv()

# Configuration
CV_DIRECTORY = "./cvs/"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_text_from_pdfs(directory: str) -> List[Dict[str, str]]:
    """
    Reads all PDF files in the directory and extracts text.
    Returns a list of dictionaries with 'source' and 'text'.
    """
    documents = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {directory}")
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            documents.append({
                "source": os.path.basename(pdf_path),
                "text": text
            })
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            
    return documents

def chunk_text(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Splits document text into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "id": f"{doc['source']}_chunk_{i}",
                "source": doc["source"],
                "text": chunk
            })
            
    print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents.")
    return chunked_docs

def run_etl():
    print("Starting ETL process...")
    
    # 1. Extract
    documents = extract_text_from_pdfs(CV_DIRECTORY)
    if not documents:
        print("No documents found or extracted. Exiting.")
        return

    # 2. Process (Chunking)
    chunks = chunk_text(documents)
    
    # 3. Embed
    embedding_model = get_embedding_model()
    print("Generating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts)
    
    # 4. Load to Pinecone
    index = get_pinecone_index(INDEX_NAME, create_if_not_exists=True)
    
    print(f"Upserting {len(chunks)} vectors to Pinecone index '{INDEX_NAME}'...")
    
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        vectors_to_upsert.append({
            "id": chunk["id"],
            "values": embeddings[i].tolist(),
            "metadata": {
                "source": chunk["source"],
                "text": chunk["text"]
            }
        })
    
    # Batch upsert (Pinecone limit is usually 100-1000 per request depending on size)
    BATCH_SIZE = 100
    for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
        batch = vectors_to_upsert[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//BATCH_SIZE + 1}/{(len(vectors_to_upsert)-1)//BATCH_SIZE + 1}")
        
    print("ETL process completed successfully.")
    
    # Verify stats
    stats = index.describe_index_stats()
    print(f"Index Stats: {stats}")

if __name__ == "__main__":
    run_etl()
