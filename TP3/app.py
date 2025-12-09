import streamlit as st
import os
from utils import get_embedding_model, get_pinecone_index, get_candidates_config
from router import execute_query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")

st.set_page_config(page_title="TP3 - Multi-Agent CV Chatbot", page_icon="🤖", layout="wide")

@st.cache_resource
def load_resources():
    """Load embedding model and Pinecone index."""
    embedding_model = get_embedding_model()
    try:
        index = get_pinecone_index(INDEX_NAME, create_if_not_exists=False)
    except Exception as e:
        st.error(f"Error connecting to Pinecone index: {e}")
        st.stop()
    return embedding_model, index

def main():
    st.title("🤖 TP3 - Multi-Agent CV Chatbot")
    st.markdown("""
    This intelligent chatbot uses **specialized agents** to answer questions about candidates.
    The system automatically detects which candidate you're asking about and routes to the appropriate agent.
    """)

    # Load resources
    with st.spinner("Loading resources..."):
        embedding_model, index = load_resources()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        debug_mode = st.checkbox("🔍 Debug Mode", value=False, help="Show routing decisions and agent details")
        
        st.markdown("---")
        st.header("📋 Available Agents")
        
        # Dynamically list agents from configuration
        candidates_config = get_candidates_config()
        for candidate_id, config in candidates_config.items():
            agent_name = config["name"].replace(" ", "") + "Agent"
            st.markdown(f"- **{agent_name}**: Answers about {config['name']}")
        st.markdown("- **CandidateAgent**: General/comparative queries")
        
        st.markdown("---")
        st.header("💡 Example Queries")
        
        # Dynamic example queries
        example_queries = []
        for config in list(candidates_config.values())[:3]:  # First 3 candidates
            name = config["name"].split()[0]  # First name
            example_queries.append(f"- \"What is {name}'s experience?\"")
        
        example_queries.extend([
            "- \"Who has Python experience?\"",
            "- \"Compare all candidates\""
        ])
        
        for query in example_queries:
            st.markdown(query)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show agent info if available
            if "agent_info" in message and message["agent_info"]:
                if debug_mode:
                    with st.expander("🔍 Agent Details"):
                        st.json(message["agent_info"])
                else:
                    st.caption(f"🤖 Handled by: **{message['agent_info'].get('agent_name', 'Unknown')}**")

    # User input
    if prompt := st.chat_input("Ask about a candidate..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt, "agent_info": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Processing with intelligent agent routing..."):
                result = execute_query(prompt, embedding_model, index, st.session_state.messages)
                
                if result["success"]:
                    response = result["response"]
                    sources = result["sources"]
                    
                    # Append sources to response
                    if sources:
                        response += f"\n\n*Sources: {', '.join(sources)}*"
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show agent information
                    agent_info = {
                        "agent_name": result["agent_name"],
                        "detected_candidate": result["detected_candidate"],
                        "sources": sources
                    }
                    
                    if debug_mode:
                        with st.expander("🔍 Agent Details"):
                            st.json(agent_info)
                    else:
                        st.caption(f"🤖 Handled by: **{result['agent_name']}**")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "agent_info": agent_info
                    })
                else:
                    error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "agent_info": None
                    })

    # Footer
    st.markdown("---")
    st.caption("TP3 - Multi-Agent System | Using LangChain agents with intelligent routing")

if __name__ == "__main__":
    main()
