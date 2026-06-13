import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🚀", layout="wide")
st.title("Zyro Dynamics HR Help Desk Chatbot")

# --- FIXED DYNAMIC DIRECTORY RESOLUTION ---
# This looks for your 'data' folder in all possible relative and absolute paths
possible_paths = ["./data", "data", "../data", "/mount/src/" + os.path.basename(os.getcwd()) + "/data"]
CORPUS_PATH = None

for path in possible_paths:
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        CORPUS_PATH = path
        break

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

@st.cache_resource
def initialize_rag_system():
    # If no path matches, gracefully inform the user instead of throwing an index out of range crash
    if CORPUS_PATH is None:
        st.error("🚨 Critical Error: The 'data' folder containing your PDF files could not be found anywhere in your GitHub repo path. Please verify that your folder is named exactly 'data' (lowercase) and contains the 11 PDFs.")
        st.stop()
        
    # Loading
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    
    if not documents:
        st.error(f"🚨 Found the directory at '{CORPUS_PATH}', but failed to read any text from the PDFs. Ensure they are valid, uncorrupted PDF documents.")
        st.stop()
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    # Vectorizing
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
    
    # Fetching API Key securely from Streamlit secrets management
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("🚨 Missing GROQ_API_KEY. Please add it to your Streamlit Advanced Secrets Settings.")
        st.stop()
        
    # LLM Models setup
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512, groq_api_key=api_key)
    
    return retriever, llm

try:
    retriever, llm = initialize_rag_system()
    st.success(f"HR Knowledge Base indexed successfully from path: '{CORPUS_PATH}'!")
except Exception as e:
    st.error(f"Error loading system pipeline: {e}")
    st.stop()

# --- CONVERSATION MEMORY & UI CHAT ENGINE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Verified Source Documents"):
                for src in msg["sources"]:
                    st.caption(f"**Source File:** {src}")

if user_query := st.chat_input("Ask a question about Zyro Dynamics HR policies:"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        # Guardrails check
        guard_prompt = ChatPromptTemplate.from_template(
            "Check scope. Respond only 'IN_SCOPE' or 'OUT_OF_SCOPE'. Question: {question}"
        )
        guard_chain = guard_prompt | llm | StrOutputParser()
        scope = guard_chain.invoke({"question": user_query}).strip()
        
        if "OUT_OF_SCOPE" in scope:
            ans_text = REFUSAL_MESSAGE
            sources_to_show = []
            st.markdown(ans_text)
        else:
            retrieved_chunks = retriever.invoke(user_query)
            context_data = "\n\n".join(c.page_content for c in retrieved_chunks)
            
            rag_prompt = ChatPromptTemplate.from_template(
                "Use context to answer. Context: {context}\nQuestion: {question}"
            )
            rag_chain_execution = (
                {"context": lambda x: context_data, "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | StrOutputParser()
            )
            ans_text = rag_chain_execution.invoke(user_query)
            st.markdown(ans_text)
            
            sources_to_show = list(set([os.path.basename(c.metadata.get('source', 'Policy Doc')) for c in retrieved_chunks]))
            if sources_to_show:
                with st.expander("View Verified Source Documents"):
                    for src in sources_to_show:
                        st.caption(f"**Source File:** {src}")
                        
        st.session_state.messages.append({
            "role": "assistant",
            "content": ans_text,
            "sources": sources_to_show
        })
