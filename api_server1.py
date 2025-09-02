# api_server1.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import os

# Import your RAG system - make sure both files are in the same directory
try:
    from Adv_Rag_chatbot import EnhancedRAGSystem
except ImportError:
    # Fallback import if running as standalone
    import sys
    sys.path.append('.')
    from Adv_Rag_chatbot import EnhancedRAGSystem

# Create app
app = FastAPI(title="RAG API Server")

# Allow OpenWebUI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama3.1:8b"

class ChatResponseChoice(BaseModel):
    message: Message
    index: int = 0
    finish_reason: str = "stop"

class ChatResponse(BaseModel):
    choices: List[ChatResponseChoice]
    model: str = "llama3.1:8b"

# Initialize RAG system
try:
    # Create documents directory if it doesn't exist
    documents_path = "./documents"
    if not os.path.exists(documents_path):
        os.makedirs(documents_path)
        print(f"📁 Created documents directory at: {documents_path}")
    
    rag_system = EnhancedRAGSystem()
    rag_system.process_documents(r"D:\Books\Gartner Predicts 2024 Ai and Automation in IT Operations.pdf")
    print("✅ RAG system initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing RAG system: {e}")
    rag_system = None

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """OpenAI-compatible chat endpoint"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        # Get the user's message
        user_message = request.messages[-1].content
        print(f"💬 User asked: {user_message}")
        
        # Get answer from RAG
        answer = rag_system.query(user_message)
        print(f"✅ Answer: {answer}")
        
        # Return in OpenAI format
        return ChatResponse(
            choices=[
                ChatResponseChoice(
                    message=Message(role="assistant", content=answer)
                )
            ]
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_initialized": rag_system is not None}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)"""
    return {
        "data": [{
            "id": "rag-chatbot",  # Unique ID for the model
            "object": "model",
            "created": 1677610602,
            "owned_by": "custom",
            "name": "My RAG Assistant",  # Display name
            "display_name": "My RAG Assistant"  # Additional display field
        }]
    }

if __name__ == "__main__":
    print("🚀 Server running on http://localhost:8001")
    print("💡 Connect OpenWebUI to: http://localhost:8001/v1")
    print("📋 Health check: http://localhost:8001/health")

    uvicorn.run(app, host="0.0.0.0", port=8001)
