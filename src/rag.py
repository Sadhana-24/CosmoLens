import requests
import json
import os
import time
import traceback
from pinecone import Pinecone
# Use the correct exception import based on your installed version (try general Exception first if unsure)
try:
    from pinecone.exceptions import ApiException
except ImportError:
    # Fallback or placeholder if specific import fails
    print("Warning: Could not import pinecone.exceptions.ApiException. Using general Exception for Pinecone errors.")
    ApiException = Exception # Use general Exception as a fallback

# Gemini Imports
import google.genai as genai
from google.genai import types

# --- Configuration ---

# Jina Embedding Configuration (Should match embedding creation)
JINA_API_URL = 'https://api.jina.ai/v1/embeddings'
JINA_MODEL = "jina-embeddings-v3"
JINA_API_KEY = os.environ.get("JINA_API_KEY", "YOUR_JINA_API_KEY") # Use environment variable or replace

# Pinecone Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY") # Use environment variable or replace
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST", "YOUR_INDEX_HOST") # Use environment variable or replace
PINECONE_NAMESPACE = "astrollava-embeddings" # Namespace used during upsert
PINECONE_TOP_K = 5 # How many relevant documents to retrieve

# Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") # Use environment variable or replace
GEMINI_MODEL = "gemini-2.0-flash" # Or another suitable Gemini model

# --- Helper Function: Get Jina Embedding for Query ---
def get_jina_embedding_for_query(query_text):
    """Gets a single embedding vector for the user query using Jina."""
    if not JINA_API_KEY or JINA_API_KEY == "YOUR_JINA_API_KEY":
        print("Error: Jina API key not configured.")
        return None

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINA_API_KEY}'
    }
    data = {
        "model": JINA_MODEL,
        "input": [query_text] # API expects a list, even for one item
    }

    try:
        response = requests.post(JINA_API_URL, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        response_data = response.json()

        # Process response - check for expected structure
        if isinstance(response_data, list) and len(response_data) > 0 and isinstance(response_data[0], dict) and 'embedding' in response_data[0]:
            return response_data[0]['embedding']
        elif isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list) and len(response_data['data']) > 0 and isinstance(response_data['data'][0], dict) and 'embedding' in response_data['data'][0]:
             return response_data['data'][0]['embedding']
        else:
            print(f"Error: Unexpected response format from Jina API: {response_data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling Jina API: {e}")
        # Check for specific status codes if needed (e.g., 401 Unauthorized, 429 Rate Limit)
        status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else "N/A"
        print(f"  Status Code: {status_code}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Jina embedding: {e}")
        traceback.print_exc()
        return None

# --- Helper Function: Search Pinecone ---
def search_pinecone(query_vector, top_k, index_host, namespace):
    """Queries Pinecone index to find similar vectors and returns their metadata."""
    if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
        print("Error: Pinecone API key not configured.")
        return []
    if not index_host or index_host == "YOUR_INDEX_HOST":
        print("Error: Pinecone index host not configured.")
        return []

    retrieved_contexts = []
    try:
        print(f"Connecting to Pinecone index: {index_host}...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(host=index_host)
        print("Querying Pinecone...")

        query_response = index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True, # Crucial to get the text back
            include_values=False   # We don't usually need the vectors themselves back
        )

        print(f"Pinecone responded with {len(query_response.matches)} matches.")
        for match in query_response.matches:
            metadata = match.metadata
            if metadata:
                # Extract the relevant text field used for context
                # Use the field you stored, likely 'combined_text_preview' or maybe 'text'/'description'
                context = metadata.get("caption", metadata.get("text", "")) # Fallback to 'text'
                if context:
                    retrieved_contexts.append(context)
                    # print(f"  - Retrieved (ID: {match.id}, Score: {match.score:.4f}): {context[:100]}...") # Debug print
                else:
                    print(f"  - Warning: Match ID {match.id} has metadata but no 'caption' or 'text' field.")
            else:
                print(f"  - Warning: Match ID {match.id} has no metadata.")

    except ApiException as e:
        print(f"Error querying Pinecone: {e}")
        # Handle specific Pinecone errors if necessary
    except Exception as e:
        print(f"An unexpected error occurred during Pinecone search: {e}")
        traceback.print_exc()

    return retrieved_contexts

# --- Helper Function: Generate Response with Gemini ---
def generate_response_with_gemini(query, context_list):
    """Generates a response using Gemini based on the query and retrieved context."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("Error: Gemini API key not configured.")
        return "Error: Gemini API key not configured."

    try:
        print("Configuring Gemini client...")
        client = genai.Client(api_key=GEMINI_API_KEY)

        # --- Construct the Prompt ---
        # Combine retrieved contexts into a single string block
        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context_list)])

        # Create a clear prompt for the RAG task
        prompt = f"""You are a helpful assistant knowledgeable about astronomy, specifically the Hubble Space Telescope findings. Answer the following user query based *only* on the provided context information. If the context does not contain the answer, state that clearly. Do not use any external knowledge.

Context Information:
---
{context_str}
---

User Query: {query}

Answer:"""

        # Prepare the contents for the Gemini API
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        # Generate content using the Gemini API
        generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

        print("Generating response with Gemini...")
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text

        return response_text.strip()

    except Exception as e:
        print(f"An error occurred during Gemini generation: {e}")
        traceback.print_exc()
        return f"Error generating response from Gemini: {e}"

# --- Main RAG Pipeline ---
def main():
    print("--- Astronomy RAG Pipeline Initialized ---")
    print("Using Jina, Pinecone, and Gemini.")

    # --- Configuration Checks ---
    if any(key in [None, "", "YOUR_JINA_API_KEY"] for key in [JINA_API_KEY]):
         print("\nError: Jina API Key is missing. Please set the JINA_API_KEY environment variable or update the script.")
         return
    if any(key in [None, "", "YOUR_PINECONE_API_KEY"] for key in [PINECONE_API_KEY]):
         print("\nError: Pinecone API Key is missing. Please set the PINECONE_API_KEY environment variable or update the script.")
         return
    if any(host in [None, "", "YOUR_INDEX_HOST"] for host in [PINECONE_INDEX_HOST]):
         print("\nError: Pinecone Index Host is missing. Please set the PINECONE_INDEX_HOST environment variable or update the script.")
         return
    if any(key in [None, "", "YOUR_GEMINI_API_KEY"] for key in [GEMINI_API_KEY]):
         print("\nError: Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable or update the script.")
         return

    while True:
        # 1. Get User Query
        print("-" * 50)
        user_query = input("Ask a question about Hubble/Astronomy (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        if not user_query.strip():
            continue

        # 2. Embed the Query
        print(f"\nEmbedding query: '{user_query[:50]}...'")
        query_embedding = get_jina_embedding_for_query(user_query)

        if query_embedding is None:
            print("Failed to get query embedding. Cannot proceed.")
            continue

        # 3. Search Pinecone
        retrieved_docs = search_pinecone(query_embedding, PINECONE_TOP_K, PINECONE_INDEX_HOST, PINECONE_NAMESPACE)

        if not retrieved_docs:
            print("Could not retrieve relevant context from Pinecone. Attempting to answer without context (may be less accurate)...")
            # Optionally, you could skip Gemini call here or proceed without context
            # For now, let's proceed but the prompt will reflect empty context.

        # 4. Generate Response using Gemini + Context
        print("\nGenerating final answer...")
        final_response = generate_response_with_gemini(user_query, retrieved_docs)

        # 5. Display Result
        print("\n--- Generated Response ---")
        print(final_response)
        print("--------------------------")

    print("\n--- RAG Pipeline Finished ---")

if __name__ == "__main__":
    # --- Remind user about dependencies ---
    print("Ensure you have installed necessary libraries:")
    print("  pip install pinecone-client google-generativeai requests")
    main()