import ollama
from app.retrieval import retrieve_chunks


def generate_answer(question: str):
    # 1. Relevant chunks retrieve karo
    retrieved_results = retrieve_chunks(question)

    # 2. Relevant information nahi mili
    if not retrieved_results:
        return {
            "question": question,
            "answer": (
                "I could not find this information "
                "in the document."
            ),
            "sources": [],
            "retrieval_distances": []
        }

    # 3. Retrieved chunks se context banao
    context = "\n\n".join(
        item["text"] for item in retrieved_results
    )

    # 4. Simple aur focused prompt
    prompt = f"""
Answer the question using the context below.

Rules:
- The context is the source of truth.
- If the answer is clearly stated, answer directly.
- Do not say information is missing when it is present.
- Do not invent information.
- Keep the answer short.

Context:
{context}

Question:
{question}

Answer:
"""

    # 5. Local Llama model se answer generate karo
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
                "temperature": 0,
                "num_predict": 100
            }
        )

        answer = response["message"]["content"].strip()

        if not answer:
            answer = (
                "The model did not generate an answer. "
                "Please try again."
            )

    except Exception as exc:
        raise RuntimeError(
            "Could not generate an answer. "
            "Check whether Ollama is running and the model "
            "llama3.2:1b is available."
        ) from exc

    # 6. Sources aur metadata return karo
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "filename": item["filename"],
                "page": item["page"],
                "text": item["text"]
            }
            for item in retrieved_results
        ],
        "retrieval_distances": [
            item["distance"]
            for item in retrieved_results
        ]
    }