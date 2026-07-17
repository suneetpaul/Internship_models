"""
Research Paper Answer Bot — Streamlit UI
Retrieves from an existing Pinecone index (built by Research_Paper_Answer_Bot.ipynb)
and answers questions with Groq's Llama model, grounded with citations.
"""

import os
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from groq import Groq

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Research Paper Answer Bot", page_icon="📚", layout="centered")

# ------------------------------------------------------------------
# Light brown theme (extra styling on top of .streamlit/config.toml)
# ------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #F5EBDD; }
 
    section[data-testid="stSidebar"] {
        background-color: #EFDFC5;
        border-right: 1px solid #D8BE93;
    }
 
    h1, h2, h3 { color: #4A2E12 !important; }
    p, span, li, label { color: #3E2A18; }
 
    /* chat bubbles */
    div[data-testid="stChatMessage"] {
        background-color: #FFF9F0;
        border: 1px solid #E1C79A;
        border-radius: 14px;
        padding: 6px 10px;
        margin-bottom: 6px;
    }
 
    /* force readable text inside chat bubbles — this is what was invisible */
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
        color: #3E2A18 !important;
    }
 
    .stCaption, [data-testid="stCaptionContainer"],
    div[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
        color: #7A5C3E !important;
    }
 
    .stButton>button {
        background-color: #8B5E3C;
        color: #FFF9F0;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover { background-color: #6E4A2E; color: #FFF9F0; }
 
    div[data-testid="stChatInput"] {
        background-color: #FFF9F0;
        border: 1px solid #D8BE93;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar — config
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    
    index_name = st.text_input("Pinecone index name", value="research-bot")
    top_k = st.slider("Chunks to retrieve (k)", 2, 10, 5)
    use_reranker = st.checkbox("Use cross-encoder re-ranker", value=True)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.caption("This app only queries the Pinecone index — run the notebook first to embed and upsert the PDFs.")
import os
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not groq_api_key or not pinecone_api_key:
    st.title("📚 Research Paper Answer Bot")
    st.info("Enter your Groq and Pinecone API keys in the sidebar to start chatting.")
    st.stop()

# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading re-ranker…")
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner=False)
def get_clients(groq_key, pine_key):
    return Groq(api_key=groq_key), Pinecone(api_key=pine_key)


embedder = load_embedder()
groq_client, pc = get_clients(groq_api_key, pinecone_api_key)

existing_indexes = [ix["name"] for ix in pc.list_indexes()]
if index_name not in existing_indexes:
    st.title("📚 Research Paper Answer Bot")
    st.error(
        f"Pinecone index '{index_name}' doesn't exist yet. "
        "Run the notebook's indexing steps first (Sections 3–6) to create and populate it."
    )
    st.stop()

index = pc.Index(index_name)
reranker = load_reranker() if use_reranker else None

GROQ_MODEL = "llama-3.1-8b-instant"
SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY the "
    "provided context. If the answer is not in the context, say you don't know. "
    "Cite sources inline as [source p.PAGE]."
)

# ------------------------------------------------------------------
# RAG pipeline (mirrors the notebook)
# ------------------------------------------------------------------
def embed(texts):
    return embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def retrieve_cosine(query, k=5):
    qv = embed([query])[0]
    res = index.query(vector=qv, top_k=k, include_metadata=True)
    return [
        {
            "text": m["metadata"]["text"],
            "source": m["metadata"]["source"],
            "page": m["metadata"]["page"],
            "score": m["score"],
        }
        for m in res["matches"]
    ]


def retrieve_rerank(query, k=5, pool=20):
    candidates = retrieve_cosine(query, k=pool)
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:k]


def build_context(hits):
    blocks = [f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits]
    return "\n\n---\n\n".join(blocks)


def rag_answer(query, k=5, use_rerank=True):
    hits = retrieve_rerank(query, k=k) if use_rerank else retrieve_cosine(query, k=k)
    context = build_context(hits)
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    answer = resp.choices[0].message.content
    sources = sorted({(h["source"], h["page"]) for h in hits})
    return answer, sources


def condense(question, history):
    if not history:
        return question
    convo = "\n".join(f"{r}: {c}" for r, c in history[-6:])
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Rewrite the follow-up question as a standalone question "
                "using the chat history. Return only the rewritten question.",
            },
            {"role": "user", "content": f"Chat history:\n{convo}\n\nFollow-up: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📚 Research Paper Answer Bot")
st.caption("Ask about the indexed papers — grounded, cited answers powered by Pinecone + Groq.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(f"{s}:p{p}" for s, p in msg["sources"]))

if prompt := st.chat_input("Ask something about the papers…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            standalone = condense(prompt, st.session_state.history)
            answer, sources = rag_answer(standalone, k=top_k, use_rerank=use_reranker)
            st.markdown(answer)
            if sources:
                st.caption("Sources: " + ", ".join(f"{s}:p{p}" for s, p in sources))

    st.session_state.history.append(("user", prompt))
    st.session_state.history.append(("assistant", answer))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
