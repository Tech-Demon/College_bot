from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import config

from loaders.web_loader import CollegeWebsiteLoader
from loaders.pdf_loader import CollegePDFLoader
from loaders.db_loader import CollegeDatabaseLoader
from vector_stores.mongodb_store import MongoDBVectorStore
from agents.bot_agent import CollegeBotAgent
from utils.helper import format_response, log_query
from langchain_core.messages import AIMessage, HumanMessage



app = FastAPI(title="College GenAI Bot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data loaders
web_loader = CollegeWebsiteLoader()
pdf_loader = CollegePDFLoader()
db_loader = CollegeDatabaseLoader()

# Initialize vector store
vector_store = MongoDBVectorStore()

# Global agent reference
college_bot_agent = None

class Query(BaseModel):
    text: str
    chat_history: Optional[List[dict]] = None

class IndexingStatus(BaseModel):
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup if collections exist."""
    try:
        web_retriever = vector_store.load_vector_store("website").as_retriever()
        pdf_retriever = vector_store.load_vector_store("pdfs").as_retriever()
        
        global college_bot_agent
        college_bot_agent = CollegeBotAgent(web_retriever, pdf_retriever, db_loader)
        print("Bot agent initialized successfully")
    except Exception as e:
        print(f"Couldn't initialize agent: {e}")
        print("You may need to run the indexing process first")

async def index_data_task():
    """Background task to index website and PDF data."""
    # Load and index website content
    print("Starting website crawling...")
    web_docs = web_loader.crawl_website()
    print(f"Crawled {len(web_docs)} website documents")
    
    web_vectordb = vector_store.create_from_documents(web_docs, "website")
    print("Website content indexed successfully")
    
    # Load and index PDF content
    print("Starting PDF indexing...")
    pdf_docs = pdf_loader.load_pdfs()
    print(f"Loaded {len(pdf_docs)} PDF documents")
    
    pdf_vectordb = vector_store.create_from_documents(pdf_docs, "pdfs")
    print("PDF content indexed successfully")
    
    # Get retrievers
    web_retriever = web_vectordb.as_retriever()
    pdf_retriever = pdf_vectordb.as_retriever()
    
    # Initialize the agent
    global college_bot_agent
    college_bot_agent = CollegeBotAgent(web_retriever, pdf_retriever, db_loader)
    print("Bot agent initialized with new data")

@app.post("/index", response_model=IndexingStatus)
async def index_data(background_tasks: BackgroundTasks):
    """Endpoint to trigger data indexing."""
    background_tasks.add_task(index_data_task)
    return {"status": "processing", "message": "Data indexing started in the background"}

@app.post("/query")
async def query_bot(query: Query):
    """Endpoint to query the GenAI bot."""
    global college_bot_agent
    
    if college_bot_agent is None:
        raise HTTPException(status_code=400, detail="Bot not initialized. Please index data first.")
    
    # Format chat history for agent
    formatted_history = []
    if query.chat_history:
        for message in query.chat_history:
            if message["role"] == "user":
                formatted_history.append(HumanMessage(content=message["content"]))
            else:
                formatted_history.append(AIMessage(content=message["content"]))
    
    # Process the query
    response = await college_bot_agent.process_query(query.text, formatted_history)
    
    # Log the interaction
    log_query(query.text, response, query.chat_history)
    
    return {"response": response}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "bot_initialized": college_bot_agent is not None,
        "mongodb_connected": True if vector_store.client else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)