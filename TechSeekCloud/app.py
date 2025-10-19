import streamlit as st
import streamlit.components.v1 as components
from groq import Groq # Replaced 'ollama' with 'groq'

# --- APP CONFIGURATION ---
st.set_page_config(page_title="TechSeek AI Agent", page_icon="🤖", layout="centered")
st.title("🤖 TechSeek AI Agent")

# --- SYSTEM PROMPT DEFINITION ---
BASE_SYSTEM_PROMPT = """
You are a senior equipment service advisor for a large equipment rental company and you've been tasked with training and supporting junior-level service technicians. 
Keep your responses concise and under 150 words. Break your responses into steps but require user input between each step and ask the user clarifying questions or follow-ups at each step when walking through diagnostics. 
Respond to the topic at a high school reading level. When possible, cite all primary sources for the information you provide. If you are unsure about an answer, respond with "I'm not sure about that. Let me look into it further." and avoid making up information.
"""

# --- GROQ CLIENT INITIALIZATION --- (MODIFIED SECTION)
try:
    # Get the API key from Streamlit secrets
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Groq API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- HELPER FUNCTIONS ---
def stream_chat(chat_messages):
    """A generator function to stream responses from Groq.""" # (MODIFIED DOCSTRING)
    try:
        # Use the Groq client to stream responses
        stream = client.chat.completions.create(
            # The model name for Llama 3 70B on Groq
            model="llama3-70b-8192", 
            messages=chat_messages,
            stream=True,
        )
        # Yield content from the streamed chunks
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")

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

# --- SIDEBAR UI ---
st.sidebar.title("Device Information")
st.session_state.make = st.sidebar.text_input("Make", value=st.session_state.make)
st.session_state.model = st.sidebar.text_input("Model", value=st.session_state.model)
st.session_state.serial_number = st.sidebar.text_input("Serial Number", value=st.session_state.serial_number)

st.sidebar.title("Actions")

# --- NEW CHAT BUTTON ---
if st.sidebar.button("✨ New Chat"):
    st.session_state.messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    st.session_state.make = ""
    st.session_state.model = ""
    st.session_state.serial_number = ""
    st.rerun()

is_chat_started = len(st.session_state.messages) > 1

# --- SAVE & PRINT BUTTONS ---
chat_text_to_save = format_chat_history(st.session_state.messages)
st.sidebar.download_button(
    label="💾 Save Chat",
    data=chat_text_to_save,
    file_name="chat_history.md",
    mime="text/markdown",
    disabled=not is_chat_started,
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