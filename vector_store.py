
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# 1. PDF load karo
loader = PyPDFLoader("documents/Bharat_RAG_Test.pdf")
documents = loader.load()


# 2. PDF text ko chunks mein divide karo
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30
)

chunks = splitter.split_documents(documents)


# 3. Local embedding model load karo
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. ChromaDB connect karo
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="bharat_documents"
)


# 5. Chunks ke embeddings generate karo
texts = [chunk.page_content for chunk in chunks]
vectors = embedding_model.encode(texts).tolist()


# 6. Chunks aur embeddings store karo
collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=texts,
    embeddings=vectors
)

print("Stored chunks:", collection.count())
print("Vector database ready!")


# 7. Interactive chatbot
print("\n===== Bharat's RAG Chatbot =====")
print("Ask questions about your PDF.")
print("Type 'exit' to quit.\n")


while True:
    question = input("You: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Chatbot: Bye bhai!")
        break

    if not question:
        continue


    # 8. Question ka embedding banao
    question_vector = embedding_model.encode(
        [question]
    )[0].tolist()


    # 9. ChromaDB se chunks retrieve karo
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=2
    )

    all_chunks = results["documents"][0]
    all_distances = results["distances"][0]


    # 10. Distance-based filtering
    distance_threshold = 1.0

    retrieved_chunks = []
    retrieved_distances = []

    for chunk, distance in zip(all_chunks, all_distances):
        if distance < distance_threshold:
            retrieved_chunks.append(chunk)
            retrieved_distances.append(distance)


    # 11. Retrieved results print karo
    print("\nDEBUG - Retrieved chunks after filtering:")

    for i, chunk in enumerate(retrieved_chunks):
        print(f"\nChunk {i + 1}")
        print(f"Distance: {retrieved_distances[i]:.4f}")
        print(chunk)


    # 12. Relevant chunk na mile toh answer generate mat karo
    if not retrieved_chunks:
        print(
            "\nChatbot: Relevant information nahi mili.\n"
        )
        continue


    # 13. Retrieved text ko context banao
    context = "\n\n".join(retrieved_chunks)


    # 14. LLM prompt
    prompt = f"""
You are a helpful assistant answering questions about a document.

Answer using only the provided context.
If the answer is not present in the context, say:
"I could not find this information in the document."
Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""


    # 15. Local Llama se answer generate karo
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # 16. Final answer
    print("\n========== CHATBOT ANSWER ==========")
    print(response["message"]["content"])
    print()
