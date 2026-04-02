import streamlit as st
import numpy as np
from openai import OpenAI
import pickle
import os

# 🔑 API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="IntelliPulse Pro AI", layout="centered")

st.title("🤖 IntelliPulse Chatbot")
st.caption("Smartwatch FAQ Bot powered by RAG")

# -------------------------------
# LOAD EMBEDDINGS (FAST)
# -------------------------------
@st.cache_resource
def load_embeddings():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)

documents = load_embeddings()

# -------------------------------
# SIMILARITY FUNCTION
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# RETRIEVE CONTEXT
# -------------------------------
def retrieve(query, top_k=3):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    scores = []

    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scores.append((score, doc["content"]))

    scores.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scores[:top_k]]

# -------------------------------
# GENERATE ANSWER (FIXED)
# -------------------------------
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are IntelliPulse Pro AI Assistant.

Instructions:
- Always interpret the user query in the context of a smartwatch
- Even if the query is vague (e.g., "I am sleeping"), relate it to smartwatch features
- Answer using ONLY the provided context
- Do NOT give generic assistant replies like "sleep well"
- Keep answer helpful and concise (3-5 lines)

Examples:
User: "I am sleeping"
→ Explain sleep tracking features

User: "I feel tired"
→ Explain health/sleep/stress tracking

Context:
{context}

User Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # Add medical disclaimer automatically
    if any(word in query.lower() for word in ["heart", "ecg", "spo2", "bp", "health", "sleep"]):
        answer += "\n\n⚠️ This is for monitoring only, not a medical diagnosis. Please consult a doctor."

    return answer

# -------------------------------
# CHAT UI (ChatGPT style)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about your smartwatch..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    context_chunks = retrieve(prompt)

    # Generate response
    answer = generate_answer(prompt, context_chunks)

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)