from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from langchain.schema.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path
import uuid
import logging
import traceback
import asyncio
import json
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define the JinaEmbeddings class (same as notebook)
class JinaEmbeddings(Embeddings):  # Updated to inherit from Embeddings
    def __init__(self, model_name=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en")):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()

# Initialize FastAPI app
app = FastAPI(title="Document QA API", description="API to answer questions based on document vector database")

# Create directories if they don't exist
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True, parents=True)
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True, parents=True)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True, parents=True)
process_status_dir = Path("process_status")
process_status_dir.mkdir(exist_ok=True, parents=True)

# Ensure directories are writable
try:
    test_file_path = uploads_dir / "test_write.txt"
    with open(test_file_path, "w") as f:
        f.write("Test write access")
    os.remove(test_file_path)
    logger.info("Upload directory is writable")
except Exception as e:
    logger.error(f"Upload directory is not writable: {str(e)}")
    raise Exception(f"Upload directory is not writable: {str(e)}")

# Set up templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set up the LLM and vector database
gemini_api_key = os.getenv("GEMINI_API_KEY")
llm_model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
temperature = float(os.getenv("TEMPERATURE", "0.0"))

# Qdrant configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "document_qa")
VECTOR_SIZE = 768  # Jina embeddings dimension

# Qdrant connection settings
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Initialize embedding model
embedding = JinaEmbeddings()

# Track if DB has been initialized
db_initialized = False
vectordb = None

# Lazy initialization of Qdrant vector database
def get_vectordb():
    global vectordb, db_initialized
    if not vectordb:
        logger.info("Initializing Qdrant vector database")
        
        # Initialize Qdrant client
        if qdrant_url and qdrant_api_key:
            # Cloud Qdrant
            logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}")
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Local Qdrant
            logger.info("Using local Qdrant instance")
            client = QdrantClient(path="./qdrant_data")
        
        # Check if collection exists, create it if not
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE
                )
            )
        
        # Initialize the Qdrant vector store with langchain
        vectordb = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embedding
        )
        
        db_initialized = True
    
    return vectordb

# Initialize LLM and QA chain
def get_qa_chain():
    # Create prompt template
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum. Keep the answer as concise as possible.

    {context}

    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Get vector database
    vectordb = get_vectordb()
    
    # Create the LLM
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, google_api_key=gemini_api_key)
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}), # Retrieve more documents
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# Text splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,  # Increased overlap for better context
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Define the request model
class QuestionRequest(BaseModel):
    question: str

# Define the response model
class AnswerResponse(BaseModel):
    answer: str
    source_documents: list = []

# Get the status of a processing job
def get_processing_status(job_id):
    status_file = process_status_dir / f"{job_id}.json"
    if not status_file.exists():
        return None
    
    try:
        with open(status_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading status file: {str(e)}")
        return None

# Update the status of a processing job
def update_processing_status(job_id, status, message=None, error=None):
    status_file = process_status_dir / f"{job_id}.json"
    
    data = {
        "job_id": job_id,
        "status": status,
        "updated_at": time.time(),
        "message": message
    }
    
    if error:
        data["error"] = str(error)
    
    try:
        with open(status_file, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Error writing status file: {str(e)}")

# Define the API endpoint for checking document processing status
@app.get("/process-status/{job_id}")
async def check_process_status(job_id: str):
    status = get_processing_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Processing job {job_id} not found")
    
    return JSONResponse(content=status)

# Define the API endpoint for asking questions
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Get the question from the request
        question = request.question
        logger.info(f"Received question: {question}")
        
        # Make sure DB is initialized
        vectordb = get_vectordb()
        
        # Check if DB has any documents
        try:
            # For Qdrant, we check collection info
            client = vectordb.client
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            count = collection_info.points_count
            if count == 0:
                logger.warning("Vector database is empty!")
                return AnswerResponse(
                    answer="I don't have any documents in my knowledge base yet. Please upload some PDF documents first.",
                    source_documents=[]
                )
            logger.info(f"Vector database has {count} documents")
        except Exception as e:
            logger.error(f"Error checking database: {str(e)}")
            return AnswerResponse(
                answer="There was an error checking the database. Please try again later.",
                source_documents=[]
            )
        
        # Get the QA chain
        qa_chain = get_qa_chain()
        
        # Query the QA chain
        result = qa_chain({"query": question})
        
        # Prepare the response
        source_docs = []
        for doc in result.get("source_documents", []):
            source_docs.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return AnswerResponse(
            answer=result["result"],
            source_documents=source_docs
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Function to process PDF in background to avoid timeout
async def process_pdf_file(job_id, file_path, original_filename):
    vectordb = None
    try:
        logger.info(f"Processing PDF in background: {file_path}, job_id: {job_id}")
        update_processing_status(job_id, "processing", "Started processing PDF")
        
        # Get or create vector database
        vectordb = get_vectordb()
        
        # Load the PDF
        update_processing_status(job_id, "processing", "Loading PDF document")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        if len(documents) == 0:
            update_processing_status(job_id, "error", "PDF has no extractable text", "No pages found in document")
            return False
        
        # Update status
        update_processing_status(job_id, "processing", f"Loaded {len(documents)} pages, splitting into chunks")
        
        # Split the documents into chunks
        chunks = text_splitter.split_documents(documents)
        if len(chunks) == 0:
            update_processing_status(job_id, "error", "Failed to extract text chunks from PDF", "No text chunks found")
            return False
            
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Add metadata to chunks to identify this document
        for i, chunk in enumerate(chunks):
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = original_filename
            chunk.metadata['chunk_id'] = i
            chunk.metadata['job_id'] = job_id
        
        # Add to vector database
        update_processing_status(job_id, "processing", f"Adding {len(chunks)} chunks to the database")
        logger.info(f"Adding {len(chunks)} chunks to vector database")
        
        # Add documents in smaller batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectordb.add_documents(batch)
            logger.info(f"Added batch {i//batch_size + 1}/{len(chunks)//batch_size + 1} to database")
        
        # Update status
        update_processing_status(job_id, "complete", f"Successfully processed PDF. Added {len(chunks)} chunks to the database.")
        logger.info(f"PDF processing complete: {original_filename}")
        return True
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in background PDF processing: {error_msg}", exc_info=True)
        update_processing_status(job_id, "error", "Error processing PDF", error_msg)
        return False

# Define the API endpoint for uploading PDF files
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    temp_file_path = None
    job_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_id = job_id
        unique_filename = f"{unique_id}{file_extension}"
        file_path = os.path.join(uploads_dir, unique_filename)
        
        # Save the uploaded file - use a more reliable approach
        temp_file_path = file_path + ".tmp"
        logger.info(f"Saving file to temporary location: {temp_file_path}")
        
        # First save to a temporary file
        try:
            with open(temp_file_path, "wb") as buffer:
                # Read in chunks to handle large files better
                CHUNK_SIZE = 1024 * 1024  # 1MB chunks
                contents = await file.read(CHUNK_SIZE)
                while contents:
                    buffer.write(contents)
                    contents = await file.read(CHUNK_SIZE)
        except Exception as e:
            logger.error(f"Error saving temporary file: {str(e)}", exc_info=True)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        # Move the temp file to the final location
        logger.info(f"Moving temporary file to final location: {file_path}")
        try:
            os.rename(temp_file_path, file_path)
        except Exception as e:
            logger.error(f"Error moving temporary file: {str(e)}", exc_info=True)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error moving file: {str(e)}")
        
        # Initialize the status file
        update_processing_status(job_id, "pending", "PDF uploaded, waiting for processing to begin")
        
        # Process the PDF in the background to avoid timeout
        logger.info("Adding PDF processing to background tasks")
        
        # Start processing in a background task to avoid timeout
        background_tasks.add_task(process_pdf_file, job_id, file_path, file.filename)
        
        return {
            "filename": file.filename, 
            "status": "success", 
            "message": "PDF uploaded successfully and processing has started in the background.",
            "job_id": job_id
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in upload_pdf: {str(e)}", exc_info=True)
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        
        error_detail = f"Error uploading PDF: {str(e)}"
        logger.error(error_detail + "\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)

# Define the home page for the UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define a health check endpoint
@app.get("/health")
async def health_check():
    # Get vector database stats
    try:
        vectordb = get_vectordb()
        client = vectordb.client
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        count = collection_info.points_count
        db_status = "ok"
    except Exception as e:
        count = 0
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "vector_db": {
            "status": db_status,
            "document_count": count
        }
    }

# GET endpoint to list all PDF files that have been uploaded
@app.get("/list-documents")
async def list_documents():
    try:
        # List all PDF files in the uploads directory
        pdf_files = []
        for file in uploads_dir.glob("*.pdf"):
            job_id = file.stem
            status = get_processing_status(job_id)
            
            pdf_files.append({
                "filename": file.name,
                "job_id": job_id,
                "size": file.stat().st_size,
                "upload_time": file.stat().st_mtime,
                "processing_status": status["status"] if status else "unknown"
            })
        
        return {"documents": pdf_files}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port) 