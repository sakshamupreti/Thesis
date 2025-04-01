
import os  # module for interacting with the operating system
from sentence_transformers import SentenceTransformer # module for sentence embeddings
import numpy as np # module for numerical operations
import faiss # module for similarity search
import requests # module for sending HTTP requests

# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directory containing the documents
docs_directory = os.path.join(script_dir, "docs")

# List all files in the directory
doc_files = [f for f in os.listdir(docs_directory) if f.endswith('.txt')]
print(f"Found files: {doc_files}")

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the contents of the documents
documents = []
for filename in doc_files:
    file_path = os.path.join(docs_directory, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        documents.append(content)
        print(f"Loaded {filename}: {content[:50]}...")
if not documents:
    print("No .txt files found in the directory.")

# Convert documents to embeddings
embeddings = model.encode(documents)

# Normalize the embeddings for cosine similarity
doc_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

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

query = "Calculate the mass flow rate (m_dot) in kg/s at the outlet of a nozzle with P_in = 200 kPa, T_in = 350 K, v_in = 50 m/s, A_in = 0.02 m^2, P_out = 150 kPa, T_out = 320 K, v_out = 120 m/s, A_out = 0.015 m^2, and R = 287 J/(kg*K) for air, verifying with inlet conditions."

#Encode the query into an embedding

query_embedding = model.encode([query])

#Normalize the query embedding 
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

k = 2  # Number of documents to retrieve

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
        response = requests.post(url, json=data)   # sends a post request to ollama's API
        response.raise_for_status()  # checks if the HTTP request was successful
        result = response.json()   # parses the json response into a dictionary
        return result.get("response", "No response received") #returns the response in the dictionary
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"


# Test it with a prompt
answer = chat_with_llama(prompt)
print("Prompt:", prompt)
print("Answer:", answer)



