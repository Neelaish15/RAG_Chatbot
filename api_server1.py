# api_server1.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import traceback

# Import your RAG system
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

# Pydantic models
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
rag_system = None
try:
    # --- KEY CHANGE ---
    # Point to the DIRECTORY containing all your documents.
    documents_directory = r"D:\Books\Gartner Predicts 2024  Ai and Automation in IT Operations.pdf"
    
    rag_system = EnhancedRAGSystem()
    # Call the new setup method just ONCE.
    rag_system.setup_from_path(documents_directory)
    
except Exception as e:
    print("❌ Error initializing RAG system")
    traceback.print_exc()
    rag_system = None

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """OpenAI-compatible chat endpoint"""
    try:
        if not rag_system or not rag_system.rag_chain:
            raise HTTPException(status_code=503, detail="RAG system is not ready or failed to initialize.")
        
        user_message = request.messages[-1].content
        print(f"💬 User asked: {user_message}")
        
        # Call RAG system
        answer = rag_system.query(user_message)
        print(f"✅ Answer: {answer}")
        
        return ChatResponse(
            choices=[ChatResponseChoice(message=Message(role="assistant", content=answer))]
        )
        
    except Exception as e:
        print("❌ Error in chat endpoint")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    is_ready = rag_system is not None and rag_system.rag_chain is not None
    return {"status": "ok" if is_ready else "initializing", "rag_initialized": is_ready}

@app.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "rag-chatbot",
            "object": "model",
            "created": 1677610602,
            "owned_by": "custom",
            "name": "My RAG Assistant",
            "display_name": "My RAG Assistant"
        }]
    }

if __name__ == "__main__":
    print("🚀 Server running on http://localhost:8001")
    print("💡 Connect OpenWebUI to: http://localhost:8001/v1")
    print("📋 Health check: http://localhost:8001/health")
    uvicorn.run(app, host="0.0.0.0", port=8001) 
