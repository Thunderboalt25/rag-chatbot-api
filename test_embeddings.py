
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "FastAPI is a Python framework for building APIs."

vector = model.encode(text)

print("Vector length:", len(vector))
print("First 5 values:", vector[:5])
