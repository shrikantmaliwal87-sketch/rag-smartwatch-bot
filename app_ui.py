import streamlit as st
import numpy as np
from openai import OpenAI
import pickle
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="IntelliPulse Pro AI", layout="centered")

st.title("🤖 IntelliPulse Chatbot")
st.caption("Smartwatch FAQ Bot powered by RAG")

# -------------------------------
# LOAD EMBEDDINGS
# -------------------------------
@st.cache_resource
def load_embeddings():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)

documents = load_embeddings()

# -------------------------------
# GREETING
# -------------------------------
def is_greeting(query):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    return query.lower().strip() in greetings

# -------------------------------
# SIMILARITY
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# RETRIEVE
# -------------------------------
def retrieve(query, top_k=5):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    scored_docs = []

    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    return scored_docs[:top_k]

# -------------------------------
# GENERATE ANSWER
# -------------------------------
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are IntelliPulse Pro AI Assistant.

- Answer using ONLY the context
- Keep it concise (3-5 lines)
- Do NOT give generic replies

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------------
# INLINE CITATION ENGINE
# -------------------------------
def add_inline_citations(answer, retrieved):
    sentences = answer.split(". ")
    final_output = ""

    for sentence in sentences:
        best_source = "Unknown"
        best_score = -1

        for score, doc in retrieved:
            similarity = cosine_similarity(
                client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[sentence]
                ).data[0].embedding,
                doc["embedding"]
            )

            if similarity > best_score:
                best_score = similarity
                best_source = doc.get("source", "Unknown")

        final_output += f"{sentence.strip()}. <span style='font-size:12px; color:gray;'>({best_source})</span> "

    return final_output

# -------------------------------
# CHAT UI
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask about your smartwatch..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if is_greeting(prompt):
        answer = "Hello 👋 I’m your IntelliPulse Pro AI Assistant. I can help you with smartwatch features, health tracking, and troubleshooting."
    else:
        retrieved = retrieve(prompt)

        context_chunks = [doc["content"] for _, doc in retrieved]

        raw_answer = generate_answer(prompt, context_chunks)

        answer = add_inline_citations(raw_answer, retrieved)

        # Add disclaimer
        if any(word in prompt.lower() for word in ["heart", "ecg", "spo2", "bp", "health", "sleep"]):
            answer += "<br><br>⚠️ This is for monitoring only, not a medical diagnosis. Please consult a doctor."

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
