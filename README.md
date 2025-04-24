# Document QA with Jina AI Embeddings

A FastAPI application that allows users to upload PDF documents, process them using Jina AI embeddings, and answer questions based on the document content.

## Features

- PDF document upload and processing
- Semantic search using Jina AI embeddings API
- Question answering based on document content using Google Gemini
- Asynchronous background processing
- Minimized memory footprint

## Requirements

- Python 3.8+
- Jina AI API key (get one at [jina.ai/embeddings](https://jina.ai/embeddings/))
- Google Gemini API key
- (Optional) Qdrant Cloud account for vector storage

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example`:
   ```
   cp .env.example .env
   ```
4. Edit the `.env` file to add your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   JINA_API_KEY=your_jina_api_key
   ```

## Running the Application

Start the application with:

```
uvicorn app:app --reload
```

Or use:

```
python app.py
```

The application will be available at http://localhost:8000

## API Endpoints

- `GET /`: Home page with UI
- `POST /upload-pdf`: Upload a PDF document
- `GET /process-status/{job_id}`: Check processing status
- `POST /ask`: Ask a question about the documents
- `GET /list-documents`: List all uploaded documents
- `GET /health`: Health check endpoint

## Memory Optimization

This application uses the Jina AI Embeddings API instead of loading embedding models locally, significantly reducing memory usage and making it suitable for deployment on platforms with memory constraints like Render.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JINA_API_KEY` | Jina AI API key | - |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `EMBEDDING_MODEL` | Jina embedding model name | `jina-embeddings-v2-base-en` |
| `LLM_MODEL` | Google Gemini model name | `gemini-2.0-flash` |
| `TEMPERATURE` | LLM temperature | `0.0` |
| `QDRANT_URL` | Qdrant cloud URL (optional) | - |
| `QDRANT_API_KEY` | Qdrant API key (optional) | - |
| `QDRANT_COLLECTION` | Qdrant collection name | `document_qa` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

## License

MIT