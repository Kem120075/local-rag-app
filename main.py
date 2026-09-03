import os
import tempfile
import warnings

# Suppress non-blocking LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------------------
# 1. MODEL CONFIGURATION
# ------------------------------------------------------------------------------
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

st.set_page_config(page_title="Local Multi-PDF RAG", layout="wide")
st.title("📚 Local Multi-PDF Chatbot with Citations")

# ------------------------------------------------------------------------------
# 2. MULTI-PDF PROCESSING & CHUNK INGESTION
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def process_uploaded_pdfs(uploaded_files):
    """Processes multiple uploaded PDF files, attaches source metadata, and indexes into ChromaDB."""
    all_documents = []

    for uploaded_file in uploaded_files:
        # Save uploaded bytes to a temporary file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Ensure metadata explicitly tracks the real file name
            for doc in docs:
                doc.metadata["file_name"] = uploaded_file.name
                
            all_documents.extend(docs)
        finally:
            os.remove(tmp_path)  # Clean up temporary local file

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    text_chunks = text_splitter.split_documents(all_documents)

    # Embed and initialize in-memory Chroma database
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )
    
    return vector_store


# ------------------------------------------------------------------------------
# 3. QA CHAIN CONSTRUCTION & RETRIEVAL LOGIC
# ------------------------------------------------------------------------------
def build_qa_chain(vector_store):
    """Builds a custom retrieval system returning generated answer and source documents."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    prompt_template = ChatPromptTemplate.from_template("""
    You are an assistant for question-answering tasks. 
    Use ONLY the following retrieved context to answer the question. 
    If you do not know the answer based on the context, say "I cannot find the answer in the document."

    Context:
    {context}

    Question: 
    {question}
    """)

    def query_rag_system(question: str):
        # Fetch top relevant context documents
        retrieved_docs = retriever.invoke(question)
        
        # Combine context text for LLM prompt
        combined_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Format prompt and invoke LLM
        prompt_val = prompt_template.format(context=combined_context, question=question)
        response = llm.invoke(prompt_val)
        
        return {
            "answer": response.content,
            "context_docs": retrieved_docs
        }

    return query_rag_system


# ------------------------------------------------------------------------------
# 4. STREAMLIT UI & CHAT INTERFACE
# ------------------------------------------------------------------------------
# Sidebar setup for Multi-PDF uploads
st.sidebar.header("Document Setup")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Embedding documents into ChromaDB..."):
        vector_store = process_uploaded_pdfs(uploaded_files)
        st.session_state.qa_chain = build_qa_chain(vector_store)
    st.sidebar.success(f"{len(uploaded_files)} PDF file(s) indexed!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_query := st.chat_input("Ask a question about your uploaded PDFs..."):
    if "qa_chain" not in st.session_state:
        st.error("Please upload at least one PDF file in the sidebar to begin.")
    else:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            result = st.session_state.qa_chain(user_query)
            answer = result["answer"]
            retrieved_docs = result["context_docs"]

            st.markdown(answer)

            # SOURCE CITATIONS TRACKING
            if retrieved_docs:
                with st.expander("📌 Source Citations"):
                    seen_sources = set()
                    for doc in retrieved_docs:
                        file_name = doc.metadata.get("file_name", "Unknown File")
                        # PyPDFLoader provides 0-indexed page numbers; convert to 1-indexed
                        page_num = doc.metadata.get("page", 0) + 1
                        
                        citation_str = f"**File:** `{file_name}` | **Page:** {page_num}"
                        if citation_str not in seen_sources:
                            st.markdown(f"- {citation_str}")
                            seen_sources.add(citation_str)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})