# pyrefly: ignore [missing-import]
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.models.models import Document
from app.schemas.schemas import DocumentOut
from app.services.ingestion import ingest_document

router = APIRouter()

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "xlsx", "csv", "md"}
ALLOWED_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/markdown",
    "text/x-markdown",
}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB


@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    domain: str = Form(default="general"),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF, .txt, .docx, .xlsx, .csv, or .md file for ingestion into the RAG system."""
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if file.content_type not in ALLOWED_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {file.content_type or 'unknown'}")

    # Verify file size before reading into memory to prevent OOM
    if hasattr(file, "size") and file.size is not None:
        file_size = file.size
    else:
        await file.seek(0, 2)
        file_size = await file.tell()
        await file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE // (1024*1024*1024)} GB.")

    file_bytes = await file.read()

    # Create DB record first (status=processing)
    doc = Document(filename=file.filename, domain=domain, status="processing")
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        chunk_count = await ingest_document(file_bytes, file.filename, domain, doc_id=doc.id)
        doc.chunk_count = chunk_count
        doc.status = "ready"
    except Exception as e:
        doc.status = "error"
        await db.commit()
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

    await db.commit()
    await db.refresh(doc)
    return doc


@router.get("/", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """List all ingested documents."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a document record. Note: does not remove chunks from ChromaDB yet."""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    await db.delete(doc)
    await db.commit()
