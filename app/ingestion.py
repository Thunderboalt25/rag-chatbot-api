from pathlib import Path
from uuid import uuid4
import hashlib

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# 1. ChromaDB initialize karo
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="bharat_documents"
)


# 2. Embedding model initialize karo
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# 3. PDF ka SHA-256 hash calculate karo
def calculate_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


# 4. PDF ingestion function
def ingest_pdf(
    pdf_path: str,
    original_filename: str | None = None
):
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            "PDF file nahi mili."
        )

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(
            "Sirf PDF files upload kar sakte hain."
        )

    filename = original_filename or pdf_file.name

    # 5. Content hash calculate karo
    file_hash = calculate_file_hash(pdf_file)

    # 6. Check karo ki same content pehle upload hua hai ya nahi
    existing = collection.get(
        where={"file_hash": file_hash},
        include=["metadatas"]
    )

    if existing["ids"]:
        existing_metadata = existing["metadatas"][0] or {}

        return {
            "message": "Duplicate PDF detected. Upload skipped.",
            "duplicate": True,
            "original_filename": existing_metadata.get(
                "filename",
                filename
            ),
            "document_id": existing_metadata.get(
                "document_id"
            ),
            "chunks_processed": 0,
            "total_chunks_in_database": collection.count()
        }

    # 7. PDF se text extract karo
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    # 8. Text ko chunks mein divide karo
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=30
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "PDF mein readable text nahi mila."
        )

    # 9. Document aur upload IDs banao
    document_id = uuid4().hex

    texts = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):
        texts.append(chunk.page_content)

        metadata = {
            "filename": filename,
            "document_id": document_id,
            "upload_id": document_id,
            "file_hash": file_hash,
            "chunk_index": index
        }

        page_number = chunk.metadata.get("page")

        if page_number is not None:
            metadata["page"] = page_number + 1

        metadatas.append(metadata)
        ids.append(f"{document_id}_{index}")

    # 10. Embeddings generate karo
    vectors = embedding_model.encode(
        texts
    ).tolist()

    # 11. Chunks aur metadata database mein save karo
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas
    )

    return {
        "message": "PDF ingested successfully.",
        "duplicate": False,
        "original_filename": filename,
        "document_id": document_id,
        "file_hash": file_hash,
        "chunks_processed": len(chunks),
        "total_chunks_in_database": collection.count()
    }