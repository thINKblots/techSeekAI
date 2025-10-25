import streamlit as st
import streamlit.components.v1 as components
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever # New import for conversation
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
# No longer need ConversationBufferMemory, we will manage history manually

# --- APP CONFIGURATION ---
st.set_page_config(page_title="TechSeek AI Agent", page_icon="🤖", layout="centered")
st.title("🤖 TechSeek AI Agent")

# --- PATHS & CONSTANTS ---
PDF_DOCS_PATH = "docs"
CHROMA_DB_PATH = "./chroma_db"
# (Your BASE_SYSTEM_PROMPT remains the same)
BASE_SYSTEM_PROMPT = """You are a senior equipment service advisor for a large equipment rental company and you've been tasked with training and supporting junior-level service technicians with troubleshooting. You have access to a knowledge base of equipment service manuals and technical documents to help you answer questions. Keep your responses concise and under 150 words and use text directly from the manuals. Break your responses into steps and require user input between them, as necessary, and ask the user clarifying questions or follow-ups at each step as necessary. Respond to the topic at a high school graduate reading level. Cite primary sources for the information you provide. If you are unsure about an answer, respond with "I'm not sure about that. My knowledge base is currently limited." and avoid making up information or gathering information from the general internet."""

# --- GROQ CLIENT INITIALIZATION ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("Groq API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- REFACTORED RAG PIPELINE SETUP FUNCTION (LCEL METHOD) ---
@st.cache_resource
def setup_rag_pipeline():
    """
    Sets up the RAG pipeline using the modern LCEL (LangChain Expression Language) approach.
    """
    # 1. Load and process documents (same as before)
    documents = []
    for file in os.listdir(PDF_DOCS_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DOCS_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=text_chunks, embedding=embeddings, persist_directory=CHROMA_DB_PATH
    )
    retriever = vector_store.as_retriever()
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # 2. Create a history-aware retriever
    # This prompt helps the LLM rephrase the user's question to be standalone, using the chat history
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. Create the main question-answering prompt
    qa_system_prompt = BASE_SYSTEM_PROMPT + """\n\nAnswer the user's question based on the context below:\n\n{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 4. Create the chain that combines documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. Create the final retrieval chain that connects everything
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- HELPER FUNCTIONS --- (Your existing function is fine)
def format_chat_history(messages):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system": continue
        role = "You" if isinstance(message, HumanMessage) else "Assistant"
        formatted_text += f"**{role}:**\n{message.content}\n\n---\n\n"
    return formatted_text

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = setup_rag_pipeline()
if "make" not in st.session_state: st.session_state.make = ""
if "model" not in st.session_state: st.session_state.model = ""
if "serial_number" not in st.session_state: st.session_state.serial_number = ""

# --- SIDEBAR UI --- (Your existing sidebar code is fine)
st.sidebar.title("Device Information")
st.session_state.make = st.sidebar.text_input("Make", value=st.session_state.make)
st.session_state.model = st.sidebar.text_input("Model", value=st.session_state.model)
st.session_state.serial_number = st.sidebar.text_input("Serial Number", value=st.session_state.serial_number)
st.sidebar.title("Actions")
if st.sidebar.button("✨ New Chat"):
    st.session_state.messages = []
    st.session_state.make = ""
    st.session_state.model = ""
    st.session_state.serial_number = ""
    # No need to re-init the chain, st.cache_resource handles it
    st.rerun()

# --- DISPLAY CHAT HISTORY ---
# Now we use the HumanMessage/AIMessage objects directly
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- MODIFIED: USER INPUT AND RAG LOGIC (LCEL METHOD) ---
if user_question := st.chat_input("Ask me a question..."):
    # Append user question as a HumanMessage object
    st.session_state.messages.append(HumanMessage(content=user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Construct device context
            device_context = ""
            if st.session_state.make: device_context += f"Make: {st.session_state.make}. "
            if st.session_state.model: device_context += f"Model: {st.session_state.model}. "
            if st.session_state.serial_number: device_context += f"Serial Number: {st.session_state.serial_number}."
            
            full_input = f"{user_question}\n\nDevice Context: {device_context}"

            # Invoke the chain with the new input structure
            result = st.session_state.conversation_chain.invoke({
                "input": full_input,
                "chat_history": st.session_state.messages
            })
            response = result['answer']
            source_documents = result.get('context', []) # 'context' holds the source docs

            st.markdown(response)

            if source_documents:
                st.markdown("---")
                st.markdown("**Sources:**")
                for doc in source_documents:
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page_number = doc.metadata.get('page', 'N/A')
                    if isinstance(page_number, int):
                        page_number += 1 # PDF pages are 0-indexed
                    st.markdown(f"- {source_file}, page {page_number}")

    # Append assistant response as an AIMessage object
    st.session_state.messages.append(AIMessage(content=response))