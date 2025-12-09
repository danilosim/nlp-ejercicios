import os
from groq import Groq
from dotenv import load_dotenv
from utils import get_candidates_config

load_dotenv()

# Configuration
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")
MODEL_ID = "llama-3.3-70b-versatile"

def get_groq_client():
    """Get Groq client for LLM calls."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=api_key)

def retrieve_context(query, embedding_model, index, top_k=3, source_filter=None):
    """
    Retrieve relevant context from Pinecone with optional source filtering.
    
    Args:
        query: The query string
        embedding_model: The embedding model to use
        index: Pinecone index
        top_k: Number of results to return
        source_filter: Optional filter for specific CV source
        
    Returns:
        tuple: (context_text, sources list)
    """
    query_embedding = embedding_model.encode(query).tolist()
    
    # Build filter if source specified
    filter_dict = None
    if source_filter:
        filter_dict = {"source": {"$eq": source_filter}}
    
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True,
        filter=filter_dict
    )
    
    contexts = []
    sources = []
    for match in results["matches"]:
        contexts.append(match["metadata"]["text"])
        sources.append(match["metadata"]["source"])
        
    return "\n\n".join(contexts), list(set(sources))

def create_candidate_agent(candidate_id: str, candidate_name: str, cv_file: str):
    """
    Factory function that creates a specialized agent for a specific candidate.
    This allows us to dynamically generate agents based on configuration.
    """
    def query_candidate(question, embedding_model, index, conversation_history=None):
        """
        Query specific to {candidate_name}'s CV.
        """
        context_text, sources = retrieve_context(
            question, 
            embedding_model, 
            index, 
            source_filter=cv_file
        )
        
        system_prompt = f"""You are a helpful assistant answering questions specifically about {candidate_name} based on their CV.
Use the following retrieved context to answer the question.
If the information is not in the context, say that you don't have that information about {candidate_name}.
Keep the answer concise and professional.

Context:
{context_text}
"""
        
        client = get_groq_client()
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-8:]:  # Last 8 messages
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = client.chat.completions.create(
            messages=messages,
            model=MODEL_ID,
            temperature=0.5,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content, sources
    
    # Set a descriptive name for the function
    query_candidate.__name__ = f"query_{candidate_id}"
    query_candidate.__doc__ = f"Query specific to {candidate_name}'s CV."
    
    return query_candidate

# Dynamically create agent functions for each candidate
_candidate_agents = {}
for candidate_id, config in get_candidates_config().items():
    _candidate_agents[candidate_id] = create_candidate_agent(
        candidate_id, 
        config["name"], 
        config["cv_file"]
    )

def get_candidate_agent(candidate_id: str):
    """Get the agent function for a specific candidate."""
    return _candidate_agents.get(candidate_id)

def query_all_candidates(question, embedding_model, index, conversation_history=None):
    """
    Query across all candidates' CVs for comparative insights.
    Dynamically includes all candidates from configuration.
    """
    candidates_config = get_candidates_config()
    
    # Retrieve context from each candidate separately
    all_contexts = []
    all_sources = []
    
    for candidate_id, config in candidates_config.items():
        context_text, sources = retrieve_context(
            question, 
            embedding_model, 
            index, 
            top_k=2, 
            source_filter=config["cv_file"]
        )
        all_contexts.append(f"--- {config['name']} ---\n{context_text}")
        all_sources.extend(sources)
    
    # Combine all contexts
    combined_context = "\n\n".join(all_contexts)
    unique_sources = list(set(all_sources))
    
    # Build candidate list for prompt
    candidate_names = [config["name"] for config in candidates_config.values()]
    candidates_str = ", ".join(candidate_names[:-1]) + f", and {candidate_names[-1]}" if len(candidate_names) > 1 else candidate_names[0]
    
    system_prompt = f"""You are a helpful assistant comparing {len(candidates_config)} candidates: {candidates_str}.
Each candidate's information is clearly labeled in the context below.
Provide fair and objective comparisons based on the question asked.
Keep the answer concise and professional.
IMPORTANT: There are exactly {len(candidates_config)} candidates, no anonymous candidates. Always name the candidates when referring to them.

Context:
{combined_context}
"""
    
    client = get_groq_client()
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if available
    if conversation_history:
        for msg in conversation_history[-8:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    response = client.chat.completions.create(
        messages=messages,
        model=MODEL_ID,
        temperature=0.5,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content, unique_sources
