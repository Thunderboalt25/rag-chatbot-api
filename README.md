# RAG Chatbot API

A document-based question-answering API built with Python, FastAPI, ChromaDB, Sentence Transformers, and Ollama.

The application retrieves relevant text chunks from uploaded PDF documents and uses a local language model to generate answers grounded in the retrieved context.

## Features

- PDF document ingestion
- Text chunking for retrieval
- Semantic search using vector embeddings
- Vector storage with ChromaDB
- Question-answering API using FastAPI
- Local LLM integration with Ollama
- Source document information in API responses
- Interactive API documentation with Swagger UI

## Tech Stack

- **Language:** Python
- **API Framework:** FastAPI
- **API Server:** Uvicorn
- **PDF Processing:** LangChain Community
- **Text Splitting:** LangChain Text Splitters
- **Embeddings:** Sentence Transformers
- **Vector Database:** ChromaDB
- **LLM Runtime:** Ollama

## How It Works

1. A PDF document is provided to the application.
2. The document text is extracted and divided into smaller chunks.
3. Sentence Transformers converts text chunks into vector embeddings.
4. ChromaDB stores the vectors and associated document metadata.
5. A user submits a question through the API.
6. The application retrieves relevant document chunks.
7. Ollama generates an answer using the retrieved context.
8. The API returns the answer and source information.

## Prerequisites

- Python installed
- Git installed
- Ollama installed and running
- The required Ollama model downloaded locally

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag_qa_system
```

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the model used by the application:

```bash
ollama pull llama3.2:1b
```

## Run the API

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open Swagger UI:

http://127.0.0.1:8000/docs

Use the available API endpoints to upload documents and ask questions.

## Example Question

```json
{
  "question": "What is Python?"
}
```

The API returns an answer, source document information, and retrieval distance values.

## Local Data

The vector database and uploaded PDF files are stored locally. They are not included in the Git repository.

After cloning the project, upload or ingest your own PDF documents to build a local knowledge base.

## Future Improvements

- Automated tests for API endpoints
- Improved retrieval evaluation
- Document management and duplicate-upload prevention
- Authentication and access control
- Deployment configuration
- Logging and monitoring

## Author

Bharat K. Mehta

GitHub: https://github.com/Thunderboalt25