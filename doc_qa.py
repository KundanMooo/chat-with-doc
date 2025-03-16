import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Custom CSS styling for a sleek dark theme
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

# Updated title and subtitle
st.title("Talk to Document")
st.caption("Made by Kundan")

# Updated prompt template: a text-based document assistant
PROMPT_TEMPLATE = """
You are a helpful text-based document assistant. Read the document provided below, summarise it concisely, and answer the user's question based on the document. 
If the answer is not found in the document, respond with: "I don't have information about that in the document."

Document:
{document_context}

User Question:
{user_query}

Answer (concise, plain text, 1-3 sentences):
"""

PERSIST_DIRECTORY = "db/chroma"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

def load_pdf_documents(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    document_loader = PDFPlumberLoader(tmp_path)
    documents = document_loader.load()
    os.remove(tmp_path)
    return documents
def list_pdf_names():
    # Retrieve all stored metadata from the persistent collection
    results = DOCUMENT_VECTOR_DB._collection.get(include=['metadatas'])
    pdf_names = set()
    # Iterate over each metadata dictionary to collect the 'pdf_name' values
    for metadata in results['metadatas']:
        if metadata and 'pdf_name' in metadata:
            pdf_names.add(metadata['pdf_name'])
    return list(pdf_names)


if st.button("Show Indexed PDF Names"):
    names = list_pdf_names()
    st.write("Indexed PDF Files:", names)


def chunk_documents(raw_documents, pdf_name):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_processor.split_documents(raw_documents)
    for chunk in chunks:
        chunk.metadata = chunk.metadata or {}
        chunk.metadata['pdf_name'] = pdf_name
    return chunks

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    DOCUMENT_VECTOR_DB.persist()

def find_related_documents(query):
    results = DOCUMENT_VECTOR_DB.similarity_search_with_relevance_scores(query, k=3)
    return [doc for doc, score in results if score > 0.35]

def generate_answer(user_query, context_documents):
    if not context_documents:
        return "This information is not mentioned in the document."
    
    context_text = "\n".join([doc.page_content for doc in context_documents])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LANGUAGE_MODEL
    response = chain.invoke({"user_query": user_query, "document_context": context_text})
    
    if "don't have information" in response.lower() and context_documents:
        return "The document contains related information but doesn't specifically address this question."
    
    return response.strip()

# UI Implementation
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
    index_documents(processed_chunks)
    st.success("✅ Document processed successfully! Ask your questions below.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
        
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
