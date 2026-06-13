import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro Dynamics HR Portal", page_icon="🚀", layout="wide")
st.title("Zyro Dynamics Enterprise HR Chatbot")

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

# --- INITIALIZE RUNTIME SESSION STATES ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# --- SIDEBAR FILE MANAGEMENT SYSTEM ---
st.sidebar.header("📁 Knowledge Base Engine")
uploaded_files = st.sidebar.file_uploader(
    "Upload the 11 corporate policy PDFs to index:", 
    type=["pdf"], 
    accept_multiple_files=True
)

@st.cache_resource
def process_uploaded_documents(upload_dir):
    """Processes uploaded PDFs and returns a pre-configured vector store retriever."""
    loader = PyPDFDirectoryLoader(upload_dir)
    documents = loader.load()
    if not documents:
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Use MMR to diversify the contexts retrieved from the dual-branding corpus files
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
    return retriever

# --- SYNC KNOWLEDGE BASE Assets ---
if uploaded_files:
    # Build or fetch cached models
    if st.session_state.retriever is None or st.session_state.llm is None:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                
        try:
            # Generate and assign to global runtime space
            st.session_state.retriever = process_uploaded_documents(temp_dir)
            
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key:
                st.sidebar.error("🚨 Missing GROQ_API_KEY in Streamlit Secrets.")
            else:
                st.session_state.llm = ChatGroq(
                    model="llama-3.1-8b-instant", 
                    temperature=0.1, 
                    max_tokens=512, 
                    groq_api_key=api_key
                )
                st.sidebar.success(f"✅ System synchronized with {len(uploaded_files)} policy assets!")
        except Exception as e:
            st.sidebar.error(f"Initialization error: {e}")
else:
    st.info("👋 Welcome! Please upload your 11 company policy PDFs in the sidebar panel to initialize the RAG Knowledge Matrix.")

# --- INTERACTIVE CHAT RUNTIME UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process input safely checking that both retriever and llm are present in state
if user_query := st.chat_input("Ask a question regarding internal employee rules:", disabled=(st.session_state.retriever is None or st.session_state.llm is None)):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Processing official guidelines..."):
            
            # Access stable session memory copies safely
            current_llm = st.session_state.llm
            current_retriever = st.session_state.retriever
            
            # Guardrails validation screening
            guard_prompt = ChatPromptTemplate.from_template(
                "Analyze scope. Respond only 'IN_SCOPE' or 'OUT_OF_SCOPE'. Consider Zyro and Acrux items in scope. Question: {question}"
            )
            guard_chain = guard_prompt | current_llm | StrOutputParser()
            scope = guard_chain.invoke({"question": user_query}).strip()
            
            if "OUT_OF_SCOPE" in scope:
                ans_text = REFUSAL_MESSAGE
                st.markdown(ans_text)
            else:
                retrieved_chunks = current_retriever.invoke(user_query)
                context_data = "\n\n".join(c.page_content for c in retrieved_chunks)
                
                rag_prompt = ChatPromptTemplate.from_template(
                    "You are the official HR Chatbot for Zyro Dynamics (Acrux Dynamics). Answer based exclusively on the context below. Context: {context}\nQuestion: {question}"
                )
                rag_chain_execution = (
                    {"context": lambda x: context_data, "question": RunnablePassthrough()}
                    | rag_prompt
                    | current_llm
                    | StrOutputParser()
                )
                ans_text = rag_chain_execution.invoke(user_query)
                st.markdown(ans_text)
                
                sources = list(set([os.path.basename(c.metadata.get('source', 'Policy Asset')) for c in retrieved_chunks]))
                if sources:
                    with st.expander("View Verified Source Documents"):
                        for src in sources:
                            st.caption(f"📌 {src}")
                            
            st.session_state.messages.append({"role": "assistant", "content": ans_text})
