from sentence_transformers import SentenceTransformer
import chromadb


# 1. Embedding model initialize karo
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# 2. ChromaDB se connect karo
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="bharat_documents"
)


# 3. Relevant chunks retrieve karo
def retrieve_chunks(
    question: str,
    n_results: int = 5,
    distance_threshold: float = 1.0
):
    question = question.strip()

    if not question:
        return []

    document_count = collection.count()

    if document_count == 0:
        return []

    # Question ka embedding generate karo
    question_vector = embedding_model.encode(
        [question]
    )[0].tolist()

    # Relevant chunks search karo
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(n_results, document_count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    chunks = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    retrieved_chunks = []
    seen_texts = set()

    for chunk, metadata, distance in zip(
        chunks,
        metadatas,
        distances
    ):
        if not chunk or not chunk.strip():
            continue

        # Distance threshold apply karo
        if distance >= distance_threshold:
            continue

        # Whitespace aur letter case normalize karke
        # exact duplicate text filter karo
        normalized_text = " ".join(
            chunk.split()
        ).casefold()

        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)

        metadata = metadata or {}

        # Original filename available ho toh use karo
        filename = (
            metadata.get("filename")
            or metadata.get("source")
            or "Legacy document (filename unavailable)"
        )

        # Page number ko safely handle karo
        page = metadata.get("page")

        retrieved_chunks.append({
            "text": chunk,
            "filename": filename,
            "page": page,
            "distance": round(float(distance), 4)
        })

    return retrieved_chunks