import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ---------------------------------------------------------
# 1. Helper Function: Get Embeddings
# ---------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    """Calls OpenAI to convert a piece of text into a vector mapping (embedding)."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ---------------------------------------------------------
# 2. Helper Function: Cosine Similarity
# ---------------------------------------------------------
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates how close two vectors are in multi-dimensional space (0 to 1)."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    print("📚 Welcome to Module 3: RAG (Retrieval-Augmented Generation)!")
    print("Loading internal knowledge base ('dummy_data.txt')...\n")
    
    # ---------------------------------------------------------
    # 3. Read and "Chunk" our Local Data
    # ---------------------------------------------------------
    try:
        with open("dummy_data.txt", "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: Could not find 'dummy_data.txt'. Please create it first.")
        return

    # In a real app we would chunk nicely (e.g. by paragraph). Here we chunk by line for simplicity.
    chunks = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
    
    # ---------------------------------------------------------
    # 4. Create the Vector Database (In-Memory)
    # ---------------------------------------------------------
    print("🧠 Embedding chunks into vectors... (This translates text to numbers)")
    vector_db = []
    for chunk in chunks:
        vector = get_embedding(chunk)
        vector_db.append({
            "text": chunk,
            "vector": vector
        })
    print("✅ Vector database ready!\n")

    # ---------------------------------------------------------
    # 5. The RAG Loop
    # ---------------------------------------------------------
    print("Ask a question about the 'AI Agents Club' (e.g., 'Who is the boss?'). Type 'exit' to quit.")
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        # Step A: Embed the user's query
        query_vector = get_embedding(user_query)
        
        # Step B: Search the Vector DB for the most similar chunks
        # We calculate the cosine similarity between the query and EVERY chunk in our "database"
        similarities = []
        for item in vector_db:
            sim = cosine_similarity(query_vector, item["vector"])
            similarities.append((sim, item["text"]))
            
        # Sort by highest similarity first and grab the top 2 results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:2]
        
        # Extract just the text of the best matches to act as our "context"
        context_texts = [result[1] for result in top_results]
        joined_context = "\n- ".join(context_texts)
        
        print(f"\n   [System found relevant context: \n- {joined_context}\n   ]")

        # Step C: Generate the answer using the retrieved context (The "Generation" in RAG)
        system_prompt = f"""
        You are a helpful assistant. Use ONLY the following retrieved context to answer the user's question.
        If the answer is not in the context, politely state that you do not know.
        
        Context:
        - {joined_context}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        
        print(f"\nAssistant: {response.choices[0].message.content}")

if __name__ == "__main__":
    main()
