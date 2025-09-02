# Adv_Rag_chatbot.py
import os
import glob
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
import time

class EnhancedRAGSystem:
    def __init__(self, persist_directory="./chroma_db", model_name="llama3.1:8b"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model="llama3.1:8b")
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Supported file extensions and their corresponding loaders
        self.loader_mapping = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
            '.pptx': UnstructuredPowerPointLoader
        }
    
    def load_documents(self, documents_path):
        """Load all supported documents from a directory"""
        documents = []
        
        # Check if path is a file or directory
        if os.path.isfile(documents_path):
            file_paths = [documents_path]
        else:
            # Get all supported files in directory
            file_paths = []
            for ext in self.loader_mapping.keys():
                file_paths.extend(glob.glob(os.path.join(documents_path, f"**/*{ext}"), recursive=True))
        
        print(f"Found {len(file_paths)} documents to process")
        
        # Load each document
        for file_path in file_paths:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self.loader_mapping:
                    loader = self.loader_mapping[ext](file_path)
                    docs = loader.load()
                    # Add source information to each document
                    for doc in docs:
                        doc.metadata['source'] = file_path
                    documents.extend(docs)
                    print(f"Loaded {file_path} with {len(docs)} chunks")
                else:
                    print(f"Skipping unsupported file type: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def process_documents(self, documents_path):
        print(f"Loading documents from: {documents_path}")

        docs = []

        if documents_path.endswith(".pdf"):
            loader = PyPDFLoader(documents_path)
            docs = loader.load()

        elif documents_path.endswith(".txt"):
            loader = TextLoader(documents_path, encoding="utf-8")
            docs = loader.load()

        elif documents_path.endswith(".docx"):
            loader = Docx2txtLoader(documents_path)
            docs = loader.load()

        elif documents_path.endswith(".ppt") or documents_path.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(documents_path)
            docs = loader.load()

        else:
            raise ValueError("Unsupported file format. Please use PDF, TXT, DOCX, or PPTX")

        # Split into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
                )
        splits = text_splitter.split_documents(docs)
                
        print(f"Split into {len(splits)} chunks")
                
                # Create vector store
        self.vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=self.embeddings,
                    persist_directory="chroma_store"
                )
                
                # Persist the vector store
        self.vectorstore.persist()
        print("Vector store created and persisted")
            
            # Create retriever
        self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Adjust based on your needs
            )
        
        # Set up the RAG chain
        self._setup_rag_chain()

    def _setup_rag_chain(self):
        """Set up the RAG chain with prompt template"""
        template = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

        Context: {context}

        Question: {question}

        Helpful Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        llm = Ollama(model="llama3.1:8b")
        
        def format_docs(docs):
            return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question):
        """Query the RAG system"""
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call process_documents first.")
        
        start_time = time.time()
        result = self.rag_chain.invoke(question)
        end_time = time.time()
        
        print(f"Query took {end_time - start_time:.2f} seconds")
        return result

    def get_document_count(self):
        """Get the number of documents in the vector store"""
        if self.vectorstore:
            return self.vectorstore._collection.count()
        return 0

# Helper function for API server
def echo_message(question: str, rag_system: EnhancedRAGSystem):
    return rag_system.query(question)

if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = EnhancedRAGSystem()
    
    # Process documents - CHANGE THIS PATH TO YOUR DOCUMENTS
    documents_path = r"D:\Books\Gartner Predicts 2024  Ai and Automation in IT Operations.pdf"
    rag_system.process_documents(documents_path)
    
    print(f"Vector store contains {rag_system.get_document_count()} chunks")
    
    # Test queries
    test_questions = [
        "What is this document about?",
        "What are the key predictions?",
        "Summarize the main points",
        "What is AI and automation in IT operations?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = rag_system.query(question)
        print(f"A: {answer}")
        print("-" * 50)
