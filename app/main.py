from pathlib import Path
from uuid import uuid4
from app.documents import router as documents_router
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)
from pydantic import BaseModel, Field

from app.ingestion import ingest_pdf
from app.rag_chain import generate_answer
from app.retrieval import collection


# 1. FastAPI Application

app = FastAPI(
    title="Bharat's RAG Chatbot API",
    description="Upload PDFs and ask questions about your documents",
    version="3.1.0"
)

app.include_router(documents_router)
# 2. Upload Directory

UPLOAD_DIR = Path("documents/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# 3. Request Model

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Ask a question about your PDF documents"
    )


# 4. Home Endpoint

@app.get("/")
def home():
    return {
        "message": "Bharat's RAG Chatbot API is running!",
        "version": "3.1.0"
    }


# 5. Health Check Endpoint

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "documents_stored": collection.count()
    }


# 6. PDF Upload and Ingestion Endpoint

@app.post("/ingest")
async def upload_pdf(file: UploadFile = File(...)):

    if (
        not file.filename
        or Path(file.filename).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    saved_filename = f"{uuid4().hex}.pdf"
    file_path = UPLOAD_DIR / saved_filename

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        file_path.write_bytes(content)

        result = ingest_pdf(
            str(file_path),
            original_filename=file.filename
        )

        # Duplicate hone par nayi temporary file delete karo
        if result.get("duplicate") is True:
            file_path.unlink(missing_ok=True)

            result["saved_filename"] = None
            result["temporary_file_deleted"] = True

        else:
            result["saved_filename"] = saved_filename
            result["temporary_file_deleted"] = False

        return result

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed."
        ) from exc

    finally:
        await file.close()


# 7. Ask Question Endpoint

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = generate_answer(question)
        return result

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred."
        ) from exc