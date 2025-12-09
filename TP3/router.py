from utils import detect_candidate, get_candidates_config
from agents import get_candidate_agent, query_all_candidates

def route_query(question: str):
    """
    Route the query to the appropriate agent based on candidate detection.
    Dynamically uses candidate configuration - scales automatically!
    
    Args:
        question: The user's query string
        
    Returns:
        tuple: (agent_function, agent_name)
    """
    detected_candidate = detect_candidate(question)
    
    # Handle general/unknown queries
    if detected_candidate in ["general", "unknown"]:
        return query_all_candidates, "CandidateAgent"
    
    # Get the specific candidate agent
    agent_func = get_candidate_agent(detected_candidate)
    if agent_func:
        # Create readable agent name from candidate_id
        candidates_config = get_candidates_config()
        candidate_name = candidates_config[detected_candidate]["name"].replace(" ", "")
        agent_name = f"{candidate_name}Agent"
        return agent_func, agent_name
    
    # Fallback to general agent
    return query_all_candidates, "CandidateAgent"

def execute_query(question: str, embedding_model, index, conversation_history=None):
    """
    Execute the routed query and return results with agent information.
    
    Args:
        question: The user's query
        embedding_model: The embedding model
        index: Pinecone index
        conversation_history: List of previous messages for context
        
    Returns:
        dict: Contains 'response', 'sources', 'agent_name', 'detected_candidate'
    """
    detected_candidate = detect_candidate(question)
    agent_func, agent_name = route_query(question)
    
    try:
        response, sources = agent_func(question, embedding_model, index, conversation_history)
        
        return {
            "response": response,
            "sources": sources,
            "agent_name": agent_name,
            "detected_candidate": detected_candidate,
            "success": True
        }
    except Exception as e:
        return {
            "response": f"Error processing query: {str(e)}",
            "sources": [],
            "agent_name": agent_name,
            "detected_candidate": detected_candidate,
            "success": False,
            "error": str(e)
        }

