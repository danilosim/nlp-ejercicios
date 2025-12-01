# CV RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions about a set of CVs.

## Setup

1.  **Install Dependencies**:
    This project uses `uv` for dependency management.
    ```bash
    cd nlp-ejercicios/TP2
    uv sync
    ```
    Or install manually:
    ```bash
    pip install -r pyproject.toml
    ```

2.  **Environment Variables**:
    Copy `.env.example` to `.env` and fill in your API keys.
    ```bash
    cp .env.example .env
    ```
    - `PINECONE_API_KEY`: Your Pinecone API key.
    - `GROQ_API_KEY`: Your Groq API key.
    - `PINECONE_INDEX_NAME`: Name for the Pinecone index (default: `cv-rag-index`).

## Usage

1.  **Ingest Data (ETL)**:
    Run the ETL script to extract text from CVs, generate embeddings, and store them in Pinecone.
    ```bash
    python etl.py
    ```
    *Note: This script reads PDFs from `../../proyectos_maestria/procesamiento_lenguaje_2/cvs/`.*

2.  **Run Chatbot**:
    Start the Streamlit application.
    ```bash
    streamlit run app.py
    ```

## Files

-   `etl.py`: Extracts text from PDFs, chunks it, and upserts vectors to Pinecone.
-   `app.py`: Streamlit chatbot interface.
-   `utils.py`: Shared utilities for Pinecone connection and embedding model loading.
