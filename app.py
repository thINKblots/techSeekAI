import streamlit as st
import streamlit.components.v1 as components
from ollama import Client, ResponseError

# ---RAG DEPENDENCIES---
from langchain_community.vectorstores import Chroma # 👈 CHANGED
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# --- APP CONFIGURATION ---
st.set_page_config(page_title="TechSeek AI Agent", page_icon="🤖", layout="centered")
st.title("🤖 TechSeek AI Agent")

# --- SYSTEM PROMPT DEFINITION ---
BASE_SYSTEM_PROMPT = """
You are a senior equipment service advisor for a large equipment rental company and you've been tasked with training and supporting junior-level service technicians. 
Keep your responses concise and under 150 words. Break your responses into steps but require user input between each step and ask the user clarifying questions or follow-ups at each step as necessary when walking through diagnostics. 
Respond to the topic at a high school reading level. When possible, cite all primary sources for the information you provide.
If you are unsure about an answer, respond with "I'm not sure about that. Let me look into it further." and avoid making up information.
"""

# --- OLLAMA CLIENT INITIALIZATION ---
try:
    client = Client()
except Exception as e:
    st.error(f"Failed to connect to Ollama. Is the Ollama server running? Error: {e}")
    st.stop()

# ---LOADING THE INDEX---
INDEX_PATH = "chroma_index" # 👈 CHANGED from FAISS

@st.cache_resource
def load_retriever():
    """
    Loads the pre-built ChromaDB index from disk and creates a retriever.
    Caches the retriever to avoid reloading on every interaction.
    """
    if not os.path.exists(INDEX_PATH):
        st.error(f"ChromaDB index not found at '{INDEX_PATH}'. Please run `build_index.py` first.")
        return None
    
    try:
        print("Loading ChromaDB index...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load the persisted database from disk
        vector_store = Chroma(
            persist_directory=INDEX_PATH, 
            embedding_function=embeddings
        )
        
        print("ChromaDB index loaded successfully.")
        return vector_store.as_retriever()
    except Exception as e:
        st.error(f"Failed to load the ChromaDB index. Error: {e}")
        return None

# --- HELPER FUNCTIONS ---
def stream_chat(chat_messages):
    """A generator function to stream responses from Ollama."""
    try:
        stream = client.chat(
            model='llama3',
            messages=chat_messages,
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']
    except ResponseError as e:
        st.error(f"Ollama model error. Is the model name correct? Details: {e.error}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def format_chat_history(messages):
    """Formats the chat history into a readable string for saving."""
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            continue
        role = "You" if message["role"] == "user" else "Assistant"
        formatted_text += f"**{role}:**\n{message['content']}\n\n---\n\n"
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
st.sidebar.title("Device Information")
st.session_state.make = st.sidebar.text_input("Make", value=st.session_state.make)
st.session_state.model = st.sidebar.text_input("Model", value=st.session_state.model)
st.session_state.serial_number = st.sidebar.text_input("Serial Number", value=st.session_state.serial_number)

st.sidebar.title("Actions")

# ---KNOWLEDGE BASE---
st.sidebar.title("📚 Knowledge Base")
if st.session_state.retriever:
    st.sidebar.success("Knowledge base loaded successfully.")
else:
    st.sidebar.error("Knowledge base not loaded. Please build the index.")

st.sidebar.title("Actions")

# --- NEW CHAT BUTTON ---
if st.sidebar.button("✨ New Chat"):
    # Reset the chat history and device info
    st.session_state.messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    st.session_state.make = ""
    st.session_state.model = ""
    st.session_state.serial_number = ""
    st.rerun()

# Check if the conversation has started
is_chat_started = len(st.session_state.messages) > 1

# Save Chat button
chat_text_to_save = format_chat_history(st.session_state.messages)
st.sidebar.download_button(
    label="💾 Save Chat",
    data=chat_text_to_save,
    file_name="chat_history.md",
    mime="text/markdown",
    disabled=not is_chat_started
)

# Print Chat button
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
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    messages_for_model = list(st.session_state.messages)
    
    device_context = ""
    if st.session_state.make: device_context += f"Make: {st.session_state.make}. "
    if st.session_state.model: device_context += f"Model: {st.session_state.model}. "
    if st.session_state.serial_number: device_context += f"Serial Number: {st.session_state.serial_number}."

    if device_context:
        full_system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nCONTEXT: The user is asking about a specific device. Use these details in your response: {device_context}"
        messages_for_model[0] = {"role": "system", "content": full_system_prompt}

    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_chat(messages_for_model))

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()