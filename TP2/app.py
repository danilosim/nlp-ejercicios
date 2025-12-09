import streamlit as st
import os
from groq import Groq
from utils import get_embedding_model, get_pinecone_index
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")
MODEL_ID = "llama-3.3-70b-versatile" # or another available Groq model

st.set_page_config(page_title="CV Chatbot RAG", page_icon="📄")

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in environment variables.")
        st.stop()
    return Groq(api_key=api_key)

@st.cache_resource
def load_resources():
    embedding_model = get_embedding_model()
    # We assume index exists now
    try:
        index = get_pinecone_index(INDEX_NAME, create_if_not_exists=False)
    except Exception as e:
        st.error(f"Error connecting to Pinecone index: {e}")
        st.stop()
    return embedding_model, index

def retrieve_context(query, embedding_model, index, top_k=6):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Organize chunks by source
    by_source = {}
    for match in results["matches"]:
        source = match["metadata"]["source"]
        text = match["metadata"]["text"]
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(text)
    
    # Build structured context
    structured_context = []
    for source, chunks in by_source.items():
        candidate_name = source.replace("_CV.txt", "").replace("_", " ")
        structured_context.append(f"--- {candidate_name} ---")
        structured_context.append("\n\n".join(chunks))
        structured_context.append("")
    
    return "\n".join(structured_context), list(by_source.keys())

def main():
    st.title("📄 TP2 -CV Chatbot (RAG)")
    st.markdown("Ask questions about the CVs in the database.")

    # Initialize resources
    with st.spinner("Loading resources..."):
        embedding_model, index = load_resources()
    
    client = get_groq_client()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Retrieve context
                context_text, sources = retrieve_context(prompt, embedding_model, index)
                
                # 2. Augment prompt
                system_prompt = f"""You are a helpful assistant that answers questions about candidates based on their CVs.
The candidates are: Danilo Reitano, Rodrigo Mesa, and Juan Garcia.
Each candidate's information is clearly labeled below.
Use the following pieces of retrieved context to answer the question. 
If you don't have information about a specific candidate, say so.
Keep the answer concise and professional.

Context:
{context_text}
"""
                
                # Construct messages with history
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add history (last 5 messages to avoid token limit)
                for msg in st.session_state.messages[-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})


                try:
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=MODEL_ID,
                        temperature=0.5,
                        max_tokens=1024,
                    )
                    response = chat_completion.choices[0].message.content
                    
                    # Append sources to response
                    if sources:
                        response += f"\n\n*Sources: {', '.join(sources)}*"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
