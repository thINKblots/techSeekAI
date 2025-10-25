import streamlit as st
import streamlit.components.v1 as components
from ollama import Client, ResponseError

# ---RAG DEPENDENCIES---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# --- APP CONFIGURATION ---
st.set_page_config(page_title="TechSeek AI Agent", page_icon="🤖", layout="centered")
st.title("🤖 TechSeek AI Agent")

# --- SYSTEM PROMPT DEFINITION ---
BASE_SYSTEM_PROMPT = """
You are a senior equipment service advisor for a large equipment rental company and you've been tasked with training and supporting junior-level service technicians with troubleshooting.
You have access to a knowledge base of equipment service manuals and technical documents to help you answer questions.
Keep your responses concise and under 150 words and use text directly from the manuals. Break your responses into steps and require user input between them, as necessary, and ask the user clarifying questions or follow-ups at each step as necessary. 
Respond to the topic at a high school graduate reading level. Cite primary sources for the information you provide.
If you are unsure about an answer, respond with "I'm not sure about that. My knowledge base is currently limited." and avoid making up information or gathering information from the general internet.
"""

# --- OLLAMA CLIENT INITIALIZATION ---
try:
    client = Client()
except Exception as e:
    st.error(f"Failed to connect to Ollama. Is the Ollama server running? Error: {e}")
    st.stop()

# ---LOADING THE INDEX---
INDEX_PATH = "chroma_index"

@st.cache_resource
def load_retriever():
    """
    Loads the pre-built ChromaDB index from disk and creates a retriever.
    Caches the retriever to avoid reloading on every interaction.
    """
    if not os.path.exists(INDEX_PATH):
        st.error(f"ChromaDB index not found at '{INDEX_PATH}'. Please run a script to build it first.")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=INDEX_PATH, embedding_function=embeddings)
        return vector_store.as_retriever()
    except Exception as e:
        st.error(f"Failed to load the ChromaDB index. Error: {e}")
        return None

# --- HELPER FUNCTIONS ---
def format_chat_history(messages):
    """
    Formats the chat history into a readable string for saving.
    This version is hardened to prevent errors from non-string content.
    """
    formatted_text = ""
    for message in messages:
        if message.get("role") == "system":
            continue
        
        content = str(message.get("content", ""))
        role = "You" if message.get("role") == "user" else "Assistant"
        formatted_text += f"**{role}:**\n{content}\n\n---\n\n"
        
    return formatted_text

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
if "make" not in st.session_state:
    st.session_state.make = ""
if "model" not in st.session_state:
    st.session_state.model = ""
if "serial_number" not in st.session_state:
    st.session_state.serial_number = ""
if "retriever" not in st.session_state:
    st.session_state.retriever = load_retriever()

# --- SIDEBAR UI ---
st.sidebar.title("Unit Information")
st.session_state.make = st.sidebar.text_input("Make", value=st.session_state.make)
st.session_state.model = st.sidebar.text_input("Model", value=st.session_state.model)
st.session_state.serial_number = st.sidebar.text_input("Serial Number", value=st.session_state.serial_number)

st.sidebar.title("📚 Knowledge Base")
if st.session_state.retriever:
    st.sidebar.success("Knowledge base loaded successfully.")
else:
    st.sidebar.error("Knowledge base not loaded.")

st.sidebar.title("Actions")
if st.sidebar.button("✨ New Chat"):
    st.session_state.messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    st.session_state.make = ""
    st.session_state.model = ""
    st.session_state.serial_number = ""
    st.rerun()

is_chat_started = len(st.session_state.messages) > 1
chat_text_to_save = format_chat_history(st.session_state.messages)

# --- CORRECTED WIDGETS ---
st.sidebar.download_button(
    label="💾 Save Chat",
    data=chat_text_to_save,
    file_name="chat_history.md",
    mime="text/markdown",
    disabled=not is_chat_started
)

disabled_attr = "disabled" if not is_chat_started else ""
disabled_style = "cursor: not-allowed; opacity: 0.5;" if not is_chat_started else ""
print_js = f"""
<button {disabled_attr} onclick="window.parent.print()">🖨️ Print Chat</button>
<style>
button {{
    display: inline-block; width: 100%; padding: 8px 16px; font-size: 16px;
    font-weight: bold; text-align: center; color: #31333F;
    background-color: #FFFFFF; border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem; cursor: pointer; {disabled_style}
}}
button:hover {{ border-color: #FF4B4B; color: #FF4B4B; }}
button:disabled:hover {{ border-color: rgba(49, 51, 63, 0.2); color: #31333F; }}
</style>
"""
components.html(print_js, height=50)

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- USER INPUT AND CHAT LOGIC ---
if user_question := st.chat_input("Ask me a question..."):
    if not st.session_state.retriever:
        st.error("Knowledge base is not loaded. Cannot answer questions.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            retrieved_docs = st.session_state.retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        with st.spinner("Thinking..."):
            device_context = ""
            if st.session_state.make: device_context += f"Make: {st.session_state.make}. "
            if st.session_state.model: device_context += f"Model: {st.session_state.model}. "
            if st.session_state.serial_number: device_context += f"Serial Number: {st.session_state.serial_number}."
            
            rag_prompt = f"""
            {BASE_SYSTEM_PROMPT}
            
            Device Context: {device_context}
            
            Use the following retrieved context from the knowledge base to answer the user's question. If the context doesn't contain the answer, say you're not sure.
            
            --- CONTEXT ---
            {context}
            --- END CONTEXT ---
            
            User Question: {user_question}
            """
            
            messages_for_model = [{"role": "system", "content": rag_prompt}]

            def stream_generator():
                """A generator that yields chunks and builds the full response."""
                full_response = ""
                stream = client.chat(model='llama3', messages=messages_for_model, stream=True)
                for chunk in stream:
                    content = chunk['message']['content']
                    full_response += content
                    yield content
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            st.write_stream(stream_generator)