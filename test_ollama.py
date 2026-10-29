import ollama

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "Explain FastAPI in two simple sentences."
        }
    ]
)

print(response["message"]["content"])
