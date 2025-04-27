# ===========================================
# Function to handle RAG: 
# Retrieve relevant chunks and generate answer
# Author: Katharine Leney, April 2025
# ===========================================

import openai
import numpy as np
import os
from assistant_config import RAG_MODEL

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def handle_query_with_rag(query, model, faiss_index, your_documents, k=5):
    # Embed the query
    query_embedding = model.encode(query)

    # Retrieve top-k relevant chunks
    D, I = faiss_index.search(np.array([query_embedding]), k)
    retrieved_chunks = [your_documents[i] for i in I[0]]

    # Build the context
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question below using only the context provided.
                If you cannot find an answer, say "Based on the provided context, there is no clear answer."

                Context:
                {context}

                Question:
                {query}

                Answer:
            """

    # Generate answer using OpenAI
    try:
        response = client.chat.completions.create(
            model=RAG_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful aviation industry assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"[Warning] Generation failed: {e}")
        answer = None

    if answer:
        return answer
    else:
        return f"[Fallback] Retrieved context:\n{context}"