# RAG Intelligence Demo

A powerful Retrieval-Augmented Generation (RAG) application that allows users to query their PDF documents using natural language. The system retrieves relevant context from the documents and generates accurate, concise answers using a Large Language Model (LLM).

## 🚀 Features

-   **Document Ingestion**: Automatically processes and chunks PDF documents.
-   **Vector Search**: Uses ChromaDB for efficient similarity search of document embeddings.
-   **Advanced RAG**: Combines retrieved context with LLM generation for high-quality answers.
-   **Modern UI**: A sleek, dark-themed web interface built with HTML/CSS/JS.
-   **Markdown Support**: Answers are rendered with rich formatting (bold, lists, code blocks).
-   **Source Attribution**: Displays the specific PDF source and page number for every answer.
-   **Confidence Score**: Shows the relevance score of the retrieved documents.
-   **File Explorer**: Sidebar listing all available PDF documents.

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript, Marked.js
-   **AI/ML**:
    -   **LangChain**: Orchestration framework.
    -   **ChromaDB**: Vector database for storing embeddings.
    -   **Sentence Transformers**: For generating text embeddings.
    -   **Groq API**: High-speed LLM inference (using `openai/gpt-oss-20b` model).
    -   **PyMuPDF**: For PDF text extraction.

## 📂 Project Structure

```
RAG Demo/
├── app.py                      # FastAPI backend application
├── data_ingestion_pipeline.py  # Script to ingest PDFs into vector store
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API Keys)
├── .gitignore                  # Git ignore rules
├── data/
│   ├── pdf_files/              # Directory to place your PDF files
│   └── vector_storage/         # Persisted ChromaDB data
├── src/
│   ├── embedding.py            # Embedding generation logic
│   ├── vector_store_chromadb.py# RAG pipeline and vector store management
│   └── document_loader.py      # PDF loading and processing
└── static/
    └── index.html              # Frontend user interface
```

## ⚡ Getting Started

### Prerequisites

-   Python 3.10+ installed.
-   [uv](https://github.com/astral-sh/uv) installed.
-   A [Groq API Key](https://console.groq.com/).

### Installation

1.  **Clone the repository** (or download the source code).
2.  **Install dependencies**:
    ```bash
    # Create a virtual environment
    uv venv

    # Activate the virtual environment
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate

    # Install dependencies
    uv pip install -r requirements.txt
    ```
3.  **Set up Environment Variables**:
    -   Create a `.env` file in the root directory.
    -   Add your Groq API key:
        ```env
        GROQ_API_KEY=your_groq_api_key_here
        ```

### 🏃‍♂️ Running the Application

#### 1. Data Ingestion (Prepare your documents)

Before asking questions, you need to process your PDF files.

1.  Place your PDF files in the `data/pdf_files/` directory.
2.  Run the ingestion pipeline:
    ```bash
    python data_ingestion_pipeline.py
    ```
    *This script will read the PDFs, generate embeddings, and store them in `data/vector_storage/`.*

#### 2. Start the Web Application

Once data is ingested, start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### 3. Access the UI

Open your browser and navigate to:

```
http://localhost:8000
```

-   **Left Sidebar**: View the list of indexed PDF files.
-   **Search Box**: Type your question and hit "Ask AI".
-   **Results**: View the AI-generated answer, confidence score, and source references.
