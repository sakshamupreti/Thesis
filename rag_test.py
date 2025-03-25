documents = [
    "The efficiency of XYZ engine is 90%",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
    "London is the capital of the United Kingdom."
]

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents to embeddings
embeddings = model.encode(documents)

# Normalize the embeddings for cosine similarity
doc_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

import faiss

#Get the dimensionality of the embeddings
#This is just to know the size of the index we need to create
dimension = doc_embeddings.shape[1]  # e.g., 384

#Create a FAISS index for inner product (cosine similarity)
#This performs a inner product calculation which results in cosine similarity
#because the vectors are normalized
index = faiss.IndexFlatIP(dimension)

#Add the document embeddings to the index
#Now the embeddings are added and the cosine similarity can be computed
index.add(doc_embeddings)

#Query the index

query = "Tell me about XYZ engine."

#Encode the query into an embedding

query_embedding = model.encode([query])

#Normalize the query embedding 
query_embedding = query_embedding / np.linalg.norm(query_embedding)

k = 5  # Number of documents to retrieve

#Search the index
"""
returns two arrays:
distances: similarity scores
indices: positions of the matching docs in the original documents list
"""
distances, indices = index.search(query_embedding, k)

#Get the retrieved documents
retrieved_docs = [documents[i] for i in indices[0]]

#Create the prompt
prompt = "Based on the following documents, answer the query.\n\nDocuments:\n"
for i, doc in enumerate(retrieved_docs):
    prompt += f"Document {i+1}: {doc}\n"
prompt += f"\nQuery: {query}\n"

import requests
import json

def chat_with_llama(prompt, model="llama3"):
    """
    Function to chat with Llama 3 using Ollama's API
    
    Args:
        prompt (str): The user prompt to send to the model
        model (str): The model to use (default is "llama3")
        
    Returns:
        str: The model's response
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Test it with a prompt
answer = chat_with_llama(prompt)
print("Prompt:", prompt)
print("Answer:", answer)
