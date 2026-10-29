
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Bharat's RAG Chatbot API",
    description="Ask questions about your PDF documents",
    version="1.1.0"
)


# ==========================================
# 2. REQUEST MODEL
# ==========================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the documents"
    )


# ==========================================
# 3. LOAD MODELS AND VECTOR DATABASE
# ==========================================

try:
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="bharat_documents"
    )

except Exception as exc:
    raise RuntimeError(
        f"Could not initialize the RAG system: {exc}"
    ) from exc


# ==========================================
# 4. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "message": "RAG Chatbot API is running!",
        "docs": "/docs",
        "ask_endpoint": "/ask"
    }


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "documents_stored": collection.count(),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama3.2:1b"
    }


# ==========================================
# 6. ASK QUESTION ENDPOINT
# ==========================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Check whether documents exist
    document_count = collection.count()

    if document_count == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge base is empty. "
                "Please ingest your PDF documents first."
            )
        )

    # --------------------------------------
    # STEP 1: CREATE QUESTION EMBEDDING
    # --------------------------------------

    try:
        question_vector = embedding_model.encode(
            [question]
        )[0].tolist()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not create the question embedding."
        ) from exc

    # --------------------------------------
    # STEP 2: RETRIEVE RELEVANT CHUNKS
    # --------------------------------------

    try:
        results = collection.query(
            query_embeddings=[question_vector],
            n_results=min(3, document_count)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document retrieval failed."
        ) from exc

    chunks = results["documents"][0] or []
    distances = results["distances"][0] or []

    # Keep only chunks that pass the current threshold.
    # This threshold is experimental and may need tuning.
    distance_threshold = 1.0

    retrieved_chunks = []
    retrieved_distances = []

    for chunk, distance in zip(chunks, distances):
        if distance < distance_threshold:
            retrieved_chunks.append(chunk)
            retrieved_distances.append(distance)

    # --------------------------------------
    # STEP 3: DEBUG INFORMATION
    # --------------------------------------

    print("\n========== RETRIEVAL DEBUG ==========")
    print("Question:", question)

    for index, (chunk, distance) in enumerate(
        zip(chunks, distances),
        start=1
    ):
        print(f"\nCandidate {index} | Distance: {distance:.4f}")
        print(chunk)

    print("\nRelevant chunks:", len(retrieved_chunks))

    # --------------------------------------
    # STEP 4: CHECK RELEVANT CONTEXT
    # --------------------------------------

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": (
                "I could not find relevant information "
                "in the document."
            ),
            "sources": [],
            "retrieval_distances": []
        }

    context = "\n\n".join(retrieved_chunks)

    # --------------------------------------
    # STEP 5: BUILD IMPROVED PROMPT
    # --------------------------------------

    prompt = f"""
You are a question-answering assistant.

Answer the user's question using the document context below.

Rules:
1. Read the entire context carefully before answering.
2. If the answer is explicitly stated in the context,
   answer directly using that information.
3. You may combine relevant statements from different
   parts of the context.
4. Do not claim that information is missing when the
   context clearly contains the answer.
5. Do not invent facts or use outside knowledge.
6. If the context does not contain the answer, respond
   exactly with:
   I could not find this information in the document.
7. Keep the answer clear and concise.

DOCUMENT CONTEXT:
<context>
{context}
</context>

USER QUESTION:
{question}

ANSWER:
"""

    # --------------------------------------
    # STEP 6: GENERATE ANSWER USING OLLAMA
    # --------------------------------------

    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        answer = response["message"]["content"].strip()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not generate an answer. "
                "Check whether Ollama is running and "
                "llama3.2:1b is available."
            )
        ) from exc

    # --------------------------------------
    # STEP 7: RETURN JSON RESPONSE
    # --------------------------------------

    print("\n========== GENERATED ANSWER ==========")
    print(answer)

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_chunks,
        "retrieval_distances": [
            round(distance, 4)
            for distance in retrieved_distances
        ]
    }
