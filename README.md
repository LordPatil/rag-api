# Document QA API

A FastAPI application that uses PDF documents and LLM (Gemini) to answer questions.

## Deployment to Render

### Prerequisites

1. A [Render](https://render.com) account
2. A [Google AI Studio](https://ai.google.dev/) API key for Gemini
3. A [Qdrant](https://qdrant.tech/) vector database (cloud or self-hosted)

### Deployment Steps

1. Fork or clone this repository
2. Log in to your Render account
3. Click on "New +" and select "Blueprint" from the dropdown menu
4. Connect your repository
5. Render will use the `render.yaml` configuration to set up your service

### Environment Variables

You need to set the following environment variables in Render:

- `GEMINI_API_KEY`: Your Google Gemini API key
- `QDRANT_URL`: URL of your Qdrant instance (if using cloud)
- `QDRANT_API_KEY`: API key for your Qdrant instance (if using cloud)

If you're using a local Qdrant instance, leave `QDRANT_URL` and `QDRANT_API_KEY` empty, and the application will default to using a local Qdrant database in the `./qdrant_data` directory.

### Persistent Storage

The application uses a persistent disk mounted at `/opt/render/project/src/uploads` to store the uploaded PDF files.

## Local Development

1. Clone the repository
2. Create a `.env` file based on `.env.example`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `uvicorn app:app --reload`

## API Endpoints

- `GET /`: Web UI for document QA
- `POST /upload-pdf`: Upload a PDF file
- `GET /process-status/{job_id}`: Check the status of PDF processing
- `POST /ask`: Ask a question about the documents
- `GET /list-documents`: List all uploaded documents
- `GET /health`: Health check endpoint

## Prerequisites

- Python 3.8+
- Chroma vector database already populated at `docs/chroma/`