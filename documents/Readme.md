# Advanced RAG Chatbot Documentation

## Overview

This project implements an Advanced Retrieval-Augmented Generation (RAG) chatbot system that can process various document types, create a vector database for efficient information retrieval, and provide intelligent responses based on the document content. The system is designed to work with OpenWebUI through a FastAPI server.

🏗️ Architecture Diagram

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   OpenWebUI     │    │   FastAPI       │    │   Ollama        │
│   (Docker)      │◄──►│   RAG Server    │◄──►│   LLM Service   │
│   Port: 3000    │    │   Port: 8001    │    │   Port: 11434   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │
│   Frontend      │    │   Document      │
│   (OpenwebUi)   │    │   Knowledge Base│
│                 │    │   ./documents/  │
│                 │    │                 │
└─────────────────┘    └─────────────────┘

## Project Structure

RAG_Bot/
├── 📄 api_server1.py              # FastAPI server (main endpoint)
├── 📄 Adv_Rag_chatbot.py          # RAG system core logic
├── 📄 requirements.txt            # Python dependencies
├── 📁 documents/                  # PDFs and other documents
│   ├── 📄 Gartner Predicts 2024 Ai and Automation in IT Operations.pdf
│   └── 📄 ...other documents...
├── 📁 chroma_db/                  # Vector database (auto-created)
└── 📁 frontend/                   # Optional Tkinter UI, inital testing for endpoints
    └── 📄 testing_UI.py

## Setup Instructions

### Prerequisites

1. Install Docker on your local machine
2. Install Ollama and pull the required model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   pip install -r requirements.txt
   ```

### OpenWebUI Setup with Docker

1. Pull and run OpenWebUI container:
   ```bash
   docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
   ```

2. Access OpenWebUI at http://localhost:3000

3. Configure OpenWebUI to connect to our API server:
   - Go to Settings → API Configuration
   - Add a new API endpoint: http://host.docker.internal:8001/v1
   - Select the model "My RAG Assistant"

## File Details

### 1. Adv_Rag_chatbot.py

This file contains the core RAG system implementation with the following features:

#### Key Components

- **EnhancedRAGSystem Class**: Main class that handles document processing and querying
- **Document Support**: PDF, TXT, DOCX, and PPTX files
- **Vector Database**: Uses ChromaDB with Ollama embeddings
- **Retrieval Mechanism**: Similarity-based retrieval with configurable parameters

#### Class Methods

- `__init__()`: Initializes the RAG system with configurable parameters
- `load_documents()`: Loads supported documents from a directory or single file
- `process_documents()`: Processes documents and creates/loads vector store
- `_setup_rag_chain()`: Sets up the RAG pipeline with prompt template
- `query()`: Queries the RAG system with a question
- `get_document_count()`: Returns the number of documents in the vector store

#### Usage Example

```python
# Initialize the RAG system
rag_system = EnhancedRAGSystem()

# Process documents (update path to your documents)
documents_path = "/path/to/your/documents"
rag_system.process_documents(documents_path)

# Query the system
response = rag_system.query("What is this document about?")
print(response)
```

### 2. api_server1.py

This file implements a FastAPI server that provides an OpenAI-compatible API for OpenWebUI integration.

#### API Endpoints

- `POST /v1/chat/completions`: Main chat endpoint for OpenWebUI
- `GET /health`: Health check endpoint
- `GET /v1/models`: Returns available models (for OpenWebUI compatibility)

#### Integration with OpenWebUI

The server is designed to work seamlessly with OpenWebUI running in Docker:
- CORS is configured to allow requests from OpenWebUI (localhost:3000)
- API responses follow OpenAI's format for compatibility
- The server runs on port 8001 to avoid conflicts with other services

#### Running the Server

```bash
python api_server1.py
```

The server will start on http://localhost:8001 and can be accessed by OpenWebUI.

### 3. requirements.txt

Contains all Python dependencies needed for the project. Key packages include:
- FastAPI: Web framework for the API server
- LangChain: Framework for building LLM applications
- ChromaDB: Vector database for document storage and retrieval
- Ollama: Python client for interacting with Ollama models
- Uvicorn: ASGI server for running FastAPI applications

## Configuration

### Document Path

Update the document path in both files to point to your documents:
- In `Adv_Rag_chatbot.py`: Update the `documents_path` variable in the `__main__` block
- In `api_server1.py`: Update the path in the `rag_system.process_documents()` call

### Model Configuration

The system uses the "llama3.1:8b" model by default. To change this:
1. Update the `model_name` parameter in the `EnhancedRAGSystem` constructor
2. Ensure the new model is available in Ollama (`ollama pull <model_name>`)

### Vector Database

The vector database is persisted in the `./chroma_db` directory. To reset the database, delete this directory.

## Usage Workflow

1. **Place Documents**: Add your documents to the specified directory
2. **Start API Server**: Run `python api_server1.py`
3. **Access OpenWebUI**: Open http://localhost:3000 in your browser
4. **Select Model**: Choose "My RAG Assistant" from the model dropdown
5. **Start Chatting**: Ask questions about your documents

## Troubleshooting

### Common Issues

1. **Ollama Connection Issues**:
   - Ensure Ollama is running: `ollama serve`
   - Verify model is downloaded: `ollama list`

2. **Document Loading Issues**:
   - Check file paths are correct
   - Verify supported file formats

3. **OpenWebUI Connection Issues**:
   - Verify API server is running on port 8001
   - Check CORS settings in the API server

4. **Vector Database Issues**:
   - Delete the `chroma_db` directory to force recreation
   - Check disk space availability

### Performance Tips

1. For large documents, adjust chunk size and overlap in `process_documents()`
2. Monitor memory usage when processing many/large documents
3. Consider using a GPU-enabled Ollama setup for better performance

## Extending the System

### Adding New Document Types

1. Add a new entry to `loader_mapping` in the `EnhancedRAGSystem` class
2. Ensure the required loader is installed (add to requirements.txt if needed)

### Customizing the Prompt

Modify the `template` variable in the `_setup_rag_chain()` method to change the system's behavior.

### Adjusting Retrieval Parameters

Modify the `search_kwargs` parameter in the `as_retriever()` call to adjust the number of retrieved documents.

## Support

For issues related to:
- Document processing: Check file formats and paths
- Model responses: Verify Ollama is working correctly
- API connectivity: Check CORS settings and ports
- OpenWebUI integration: Verify model configuration in OpenWebUI

**End of the Document**
