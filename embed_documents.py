
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. PDF load karo
loader = PyPDFLoader("documents/Bharat_RAG_Test.pdf")
documents = loader.load()

# 2. Text ko chunks mein split karo
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# 3. Local embedding model load karo
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Har chunk ka embedding generate karo
texts = [chunk.page_content for chunk in chunks]
vectors = model.encode(texts)

print("Total chunks:", len(chunks))
print("Embeddings shape:", vectors.shape)

for i, vector in enumerate(vectors):
    print(f"Chunk {i + 1}: vector length = {len(vector)}")

from sklearn.metrics.pairwise import cosine_similarity

# User question
question = "What is FastAPI?"

# Question ko vector mein convert karo
question_vector = model.encode([question])

# Question aur document chunks compare karo
scores = cosine_similarity(question_vector, vectors)[0]

# Sabse relevant chunk ka index
best_index = scores.argmax()

print("\nQuestion:", question)
print("Similarity scores:", scores)
print("\nMost relevant chunk:")
print(chunks[best_index].page_content)
