import os
import json
import numpy as np
import faiss
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ENV Vars
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")                        # Chat deployment name
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING")  # Embedding deployment name
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")               # e.g. 2023-05-15

# Azure URLs
AZURE_EMBEDDING_URL = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_OPENAI_VERSION}"
AZURE_CHAT_URL = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_VERSION}"

# Request headers
HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}

# Load FAISS index
index = faiss.read_index("embeddings/faiss_index.idx")

# Load metadata
with open("embeddings/metadata.json", encoding="utf-8") as f:
    questions = json.load(f)

# Load raw faqs for answers
with open("data/faqs.json", encoding="utf-8") as f:
    raw_faqs = json.load(f)

# Flatten raw_faqs into indexed list
faqs = []
for category, items in raw_faqs.items():
    for item in items:
        faqs.append({
            "question": item["question"],
            "answer": item["answer"],
            "category": category
        })

# ========== Embedding Function ==========
def get_embedding(text):
    body = { "input": text }
    response = requests.post(AZURE_EMBEDDING_URL, headers=HEADERS, json=body)
    if response.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {response.text}")
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

# ========== Chat Completion Function ==========
def get_chat_completion(prompt):
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful support assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 512
    }
    response = requests.post(AZURE_CHAT_URL, headers=HEADERS, json=body)
    if response.status_code != 200:
        raise RuntimeError(f"Chat request failed: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

# ========== Retrieval Function ==========
def retrieve(query, k=3):
    emb = get_embedding(query)
    distances, ids = index.search(emb.reshape(1, -1), k)
    return [(questions[i]["question"], float(distances[0][j]), i) for j, i in enumerate(ids[0])]

# ========== Final Answer Function ==========
def answer(query, lang="en"):
    candidates = retrieve(query)
    top_q, dist, idx = candidates[0]

    if dist > 0.5:
        return {
            "answer": "I’m not confident I know the answer to that. Could you please rephrase or ask something else?",
            "source": None
        }

    prompt = (
        f"User question: {query}\n\n"
        f"Known Q&A: {top_q} → {faqs[idx]['answer']}\n\n"
        f"Please answer in a friendly and clear tone in this language: {lang}."
    )

    reply = get_chat_completion(prompt)

    return {
        "answer": reply.strip(),
        "source": top_q
    }
