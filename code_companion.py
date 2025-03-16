import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling for a dark theme
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Configuration
MAX_HISTORY = 6
TEMPERATURE = 0.5
MODEL_CONTEXT_SIZE = 4096

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize the chat engine
@st.cache_resource
def get_llm_engine(model):
    return ChatOllama(
        model=model,
        base_url="http://localhost:11434",
        temperature=TEMPERATURE,
        top_p=0.9,
        repeat_penalty=1.1
    )

llm_engine = get_llm_engine(selected_model)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English. "
    "If unsure, ask clarifying questions. Format code with markdown."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

def trim_history(messages, max_tokens=MODEL_CONTEXT_SIZE * 0.7):
    current_length = sum(len(msg["content"]) for msg in messages)
    while current_length > max_tokens and len(messages) > 1:
        removed = messages.pop(1)
        current_length -= len(removed["content"])
    return messages

# Chat interface
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_query = st.chat_input("Type your coding question here...")

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    history_messages = st.session_state.message_log[-MAX_HISTORY*2:]
    for msg in history_messages:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_response(prompt_chain):
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    st.session_state.message_log = trim_history(st.session_state.message_log)
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_response(prompt_chain)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.session_state.message_log = trim_history(st.session_state.message_log)
    
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(ai_response)
