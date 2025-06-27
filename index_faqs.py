import json
import os
import requests
import numpy as np
import faiss
from dotenv import load_dotenv

# Load env
load_dotenv()

# Azure credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://mahindra-openai.openai.azure.com/
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")    # secret key
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING")  # e.g. text-embedding-ada-002
API_VERSION = "2023-05-15"  # Recommended stable API version

# Azure endpoint for embeddings
EMBEDDING_URL = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/embeddings?api-version={API_VERSION}"

# Request headers
HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY
}

# Load and flatten FAQs
with open("data/faqs.json", "r", encoding="utf-8") as f:
    raw_faqs = json.load(f)

faqs = []
for category, items in raw_faqs.items():
    for item in items:
        faqs.append({
            "question": item["question"],
            "answer": item["answer"],
            "category": category
        })

# Embedding function using REST API
def get_embedding(text: str):
    body = {
        "input": text
    }
    response = requests.post(EMBEDDING_URL, headers=HEADERS, json=body)
    if response.status_code != 200:
        print("❌ Failed:", response.text)
        raise Exception("Embedding request failed")
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

# Build embeddings
print("🔄 Generating embeddings...")
questions = [faq["question"] for faq in faqs]
embeddings = np.vstack([get_embedding(q) for q in questions])

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss_index.idx")

with open("embeddings/metadata.json", "w", encoding="utf-8") as f:
    json.dump(faqs, f, indent=2, ensure_ascii=False)

print(f"✅ Indexed {len(faqs)} questions and saved FAISS index.")
