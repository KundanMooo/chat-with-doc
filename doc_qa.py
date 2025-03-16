import os
import sqlite3
import json
import tempfile
import streamlit as st
import numpy as np

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# -------------------- Custom CSS for Dark Theme --------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- App Title --------------------
st.title("Talk to Document")
st.caption("Made by Kundan")

# -------------------- Prompt Template --------------------
# This prompt instructs the LLM to only answer based on the document.
PROMPT_TEMPLATE = (
    "You are a helpful text-based document assistant. Answer the following question "
    "strictly using only the information provided in the document context below. "
    "If the answer is not found in the document, respond with: "
    "\"I don't have information about that in the document.\"\n\n"
    "Document Context:\n{document_context}\n\n"
    "User Question:\n{user_query}\n\n"
    "Answer (concise, plain text, 1-3 sentences):"
)

# -------------------- Initialize Models --------------------
# Using the Ollama model "deepseek-r1:1.5b" for both embeddings and LLM
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# -------------------- SQLite Database Setup --------------------
try:
    cache_decorator = st.cache_resource
except AttributeError:
    cache_decorator = st.cache(allow_output_mutation=True)

@cache_decorator
def get_connection():
    # Create (or connect to) a SQLite database file
    conn = sqlite3.connect("documents.db", check_same_thread=False)
    return conn

def init_db():
    conn = get_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_name TEXT,
        chunk_id INTEGER,
        content TEXT,
        embedding TEXT
    )
    ''')
    conn.commit()

init_db()  # Create the table on startup

# -------------------- PDF Loading & Processing --------------------
def load_pdf_documents(uploaded_file):
    """Save the uploaded PDF temporarily and extract its text."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    document_loader = PDFPlumberLoader(tmp_path)
    documents = document_loader.load()
    os.remove(tmp_path)
    return documents

def chunk_documents(raw_documents, pdf_name):
    """
    Split documents into smaller chunks.
    Each chunk will later be embedded and stored with its PDF name.
    """
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_processor.split_documents(raw_documents)
    # Attach the pdf_name in each chunk's metadata for later reference.
    for chunk in chunks:
        chunk.metadata = chunk.metadata or {}
        chunk.metadata['pdf_name'] = pdf_name
    return chunks

def index_documents(document_chunks, pdf_name):
    """
    For each document chunk, compute its embedding and store it in the SQLite table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    for i, chunk in enumerate(document_chunks):
        # Compute embedding for the chunk's text. embed_documents returns a list.
        embedding_vector = EMBEDDING_MODEL.embed_documents([chunk.page_content])[0]
        embedding_str = json.dumps(embedding_vector)  # Store as JSON string
        cursor.execute(
            "INSERT INTO document_chunks (pdf_name, chunk_id, content, embedding) VALUES (?, ?, ?, ?)",
            (pdf_name, i, chunk.page_content, embedding_str)
        )
    conn.commit()

def list_pdf_names():
    """Retrieve all unique PDF names that have been indexed."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT pdf_name FROM document_chunks")
    rows = cursor.fetchall()
    return [row[0] for row in rows]

# -------------------- Similarity Search Functions --------------------
def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_related_documents(query, k=3, threshold=0.35):
    """
    Given a user query, embed it and compare with all stored embeddings.
    Return the top k chunks (texts) whose cosine similarity is above the threshold.
    """
    query_embedding = EMBEDDING_MODEL.embed_documents([query])[0]
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content, embedding FROM document_chunks")
    rows = cursor.fetchall()
    similarities = []
    for content, embedding_str in rows:
        stored_embedding = json.loads(embedding_str)
        sim = cosine_similarity(query_embedding, stored_embedding)
        similarities.append((sim, content))
    filtered = [ (sim, content) for sim, content in similarities if sim > threshold ]
    filtered.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [content for sim, content in filtered[:k]]
    return top_chunks

# -------------------- LLM Answer Generation --------------------
def generate_answer(user_query, context_documents):
    if not context_documents:
        return "This information is not mentioned in the document."
    
    # Combine context from the most relevant chunks.
    context_text = "\n".join(context_documents)
    # Format the prompt with the document context and user question.
    prompt_text = PROMPT_TEMPLATE.format(document_context=context_text, user_query=user_query)
    
    # Pass the prompt text directly to the LLM.
    response = LANGUAGE_MODEL(prompt_text)
    
    # If the LLM returns the default message, adjust the answer.
    if "don't have information" in response.lower():
        return "The document contains related information but doesn't specifically address this question."
    
    return response.strip()

# -------------------- UI Implementation --------------------
st.markdown("---")

uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    raw_docs = load_pdf_documents(uploaded_pdf)
    processed_chunks = chunk_documents(raw_docs, uploaded_pdf.name)
    index_documents(processed_chunks, uploaded_pdf.name)
    st.success("✅ Document processed and indexed successfully! Ask your questions below.")

    # Show previously indexed PDF names if needed.
    if st.button("Show Indexed PDF Names"):
        names = list_pdf_names()
        st.write("Indexed PDF Files:", names)

    # Set up a chat history in session state.
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat messages.
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # Chat input for user question.
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Find related document chunks from our database.
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
        
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
