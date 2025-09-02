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
from langchain_ollama import OllamaEmbeddings
import time

class EnhancedRAGSystem:
    def __init__(self, persist_directory="./chroma_store", model_name="llama3.1:8b"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=self.model_name)
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
    
    def _load_documents_from_path(self, documents_path):
        """Internal method to load all supported documents from a file or directory"""
        all_docs = []
        
        # Determine if it's a file or a directory
        if os.path.isfile(documents_path):
            file_paths = [documents_path]
        elif os.path.isdir(documents_path):
            file_paths = []
            for ext in self.loader_mapping.keys():
                file_paths.extend(glob.glob(os.path.join(documents_path, f"**/*{ext}"), recursive=True))
        else:
            print(f"Error: Path {documents_path} is not a valid file or directory.")
            return []

        print(f"Found {len(file_paths)} documents to process.")
        
        # Load each document
        for file_path in file_paths:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self.loader_mapping:
                    loader = self.loader_mapping[ext](file_path)
                    docs = loader.load()
                    # Add source information to each document
                    for doc in docs:
                        doc.metadata['source'] = os.path.basename(file_path)
                    all_docs.extend(docs)
                    print(f"-> Loaded {len(docs)} pages/chunks from {os.path.basename(file_path)}")
                else:
                    print(f"Skipping unsupported file type: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        return all_docs

    def setup_from_path(self, documents_path):
        """Loads, splits, and indexes documents from a given path to build the RAG system."""
        print(f"Setting up RAG system from path: {documents_path}")
        
        # Step 1: Load all documents
        docs = self._load_documents_from_path(documents_path)
        if not docs:
            print("No documents were loaded. RAG system setup aborted.")
            return

        # Step 2: Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Split {len(docs)} documents into {len(splits)} chunks.")
        
        # Step 3: Create and persist the vector store from all chunks at once
        print("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print("Vector store created and persisted successfully.")
        
        # Step 4: Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        # Step 5: Set up the RAG chain
        self._setup_rag_chain()
        print("✅ RAG chain is ready.")

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
        llm = Ollama(model=self.model_name)
        
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
            raise ValueError("RAG chain not initialized. Call setup_from_path first.")
        
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

# Main block for direct testing
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = EnhancedRAGSystem()
    
    # Process documents from a DIRECTORY - CHANGE THIS PATH
    documents_directory = r"D:\Books\Gartner Predicts 2024  Ai and Automation in IT Operations.pdf" # <-- Point to a folder containing your files
    rag_system.setup_from_path(documents_directory)
    
    print(f"Vector store contains {rag_system.get_document_count()} chunks")
    
    # Test queries
    test_questions = [
        "What is this document about?",
        "What are the key predictions from Gartner?",
        "Explain AI-EV in simple terms",
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = rag_system.query(question)
        print(f"A: {answer}")
        print("-" * 50)
