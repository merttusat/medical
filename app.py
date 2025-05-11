# app.py
import streamlit as st
import torch
import requests
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# ——— AYARLAR ———
ES_URL        = "http://localhost:9200"
INDEX_NAME    = "pdf_greenfield"
TOP_K         = 5
OLLAMA_URL    = "http://localhost:11434"
OLLAMA_MODEL  = "gemma3:1b"

# Initialize clients/models once
es = Elasticsearch(ES_URL)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    device=device,
    trust_remote_code=True
)

def is_medical_query(prompt: str) -> bool:
    # Zero-shot English/Turkish yes/no
    system_msg = (
        "You are a classification assistant specialized in medical topics. "
        "Is the following question about medical or healthcare topics? "
        "Answer with exactly Yes or No."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role":"system",  "content": system_msg},
            {"role":"user",    "content": prompt},
        ],
        "max_new_tokens": 4,
        "temperature": 0.0,
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload,
                      headers={"Content-Type":"application/json"})
    r.raise_for_status()
    ans = r.json()["choices"][0]["message"]["content"].strip().lower()
    return ans.startswith("y")

def retrieve_chunks(query: str, top_k: int = TOP_K):
    q_vec = embedder.encode([query], convert_to_tensor=True)[0].to("cpu").tolist()
    body = {
        "size": top_k,
        "_source": ["chunk"],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.q,'embedding')+1.0",
                    "params": {"q": q_vec}
                }
            }
        }
    }
    resp = es.search(index=INDEX_NAME, body=body)
    hits = resp["hits"]["hits"]
    ids = [int(h["_id"]) for h in hits]
    chunks = [h["_source"]["chunk"] for h in hits]

    # komşu chunk’lar
    neigh = {i+1 for i in ids} | {i-1 for i in ids}
    neigh -= set(ids)
    if neigh:
        m = es.mget(index=INDEX_NAME, body={"ids": list(neigh)})
        for doc in m["docs"]:
            if doc.get("found"):
                chunks.append(doc["_source"]["chunk"])
    return chunks

def ask_llm(prompt: str) -> str:
    system_msg = (
        "You are an expert medical assistant. "
        "Answer only using the provided context. "
        "If answer is not in context, reply 'Üzgünüm, yeterli bilgi bulamadım.'"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role":"system", "content": system_msg},
            {"role":"user",   "content": prompt}
        ],
        "max_new_tokens": 512,
        "temperature": 0.2,
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload,
                      headers={"Content-Type":"application/json"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def build_rag_prompt(query: str, contexts: list[str]) -> str:
    header = "Use the following text snippets to answer the question:\n\n"
    body   = "\n\n---\n\n".join(contexts)
    return f"{header}{body}\n\nQuestion: {query}\nAnswer:"

# ——— Streamlit UI ———
st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")
st.title("🩺 Medical RAG Chatbot")

query = st.text_input("Enter your question:", "")
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif not is_medical_query(query):
        st.error("This question is not medical/healthcare related.")
    else:
        with st.spinner("Retrieving context..."):
            contexts = retrieve_chunks(query)
        if not contexts:
            st.error("Üzgünüm, yeterli bilgi bulamadım.")
        else:
            st.subheader("Retrieved Context Snippets")
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**[{i}]** {c}")
            rag_prompt = build_rag_prompt(query, contexts)
            with st.spinner("Generating answer..."):
                answer = ask_llm(rag_prompt)
            st.subheader("Answer")
            st.write(answer)
