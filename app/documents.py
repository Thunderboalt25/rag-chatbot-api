from fastapi import APIRouter, HTTPException

from app.retrieval import collection


# 1. Router initialize karo
router = APIRouter(
    prefix="/documents",
    tags=["Document Management"]
)


# 2. GET /documents
# Saare unique documents ki list return karega

@router.get("/")
def list_documents():

    result = collection.get(
        include=["metadatas"]
    )

    documents = {}

    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])

    for chunk_id, metadata in zip(ids, metadatas):

        metadata = metadata or {}

        document_id = metadata.get("document_id")

        # Purane records ko safely skip karo
        if not document_id:
            continue

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata.get(
                    "filename",
                    "Unknown document"
                ),
                "file_hash": metadata.get("file_hash"),
                "total_chunks": 0
            }

        documents[document_id]["total_chunks"] += 1

    return {
        "total_documents": len(documents),
        "documents": list(documents.values())
    }


# 3. GET /documents/{document_id}
# Ek document ki details return karega

@router.get("/{document_id}")
def get_document(document_id: str):

    result = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"]
    )

    ids = result.get("ids", [])

    if not ids:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    metadatas = result.get("metadatas", [])
    texts = result.get("documents", [])

    metadata = metadatas[0] or {}

    chunks = []

    for chunk_id, chunk_text, chunk_metadata in zip(
        ids,
        texts,
        metadatas
    ):
        chunk_metadata = chunk_metadata or {}

        chunks.append({
            "chunk_id": chunk_id,
            "page": chunk_metadata.get("page"),
            "chunk_index": chunk_metadata.get("chunk_index"),
            "text": chunk_text
        })

    return {
        "document_id": document_id,
        "filename": metadata.get(
            "filename",
            "Unknown document"
        ),
        "file_hash": metadata.get("file_hash"),
        "total_chunks": len(chunks),
        "chunks": chunks
    }


# 4. DELETE /documents/{document_id}
# Document ke saare chunks delete karega

@router.delete("/{document_id}")
def delete_document(document_id: str):

    # Pehle verify karo ki document exist karta hai
    existing = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    chunks_deleted = len(existing["ids"])

    # Sirf isi document ke chunks delete karo
    collection.delete(
        where={"document_id": document_id}
    )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "chunks_deleted": chunks_deleted,
        "total_chunks_in_database": collection.count()
    }