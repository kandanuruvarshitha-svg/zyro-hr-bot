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

# --- SIDEBAR FILE MANAGEMENT SYSTEM ---
st.sidebar.header("📁 Knowledge Base Engine")
uploaded_files = st.sidebar.file_uploader(
    "Upload the 11 corporate policy PDFs to index:", 
    type=["pdf"], 
    accept_multiple_files=True
)

@st.cache_resource
def initialize_rag_system(upload_dir):
    loader = PyPDFDirectoryLoader(upload_dir)
    documents = loader.load()
    if not documents:
        return None, None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
    
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512, groq_api_key=api_key)
    
    return retriever, llm

retriever, llm = None, None

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            
    try:
        retriever, llm = initialize_rag_system(temp_dir)
        if retriever and llm:
            st.sidebar.success(f"✅ Knowledge base synchronized with {len(uploaded_files)} policy assets!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
else:
    st.info("👋 Welcome! Please upload your 11 company policy PDFs in the sidebar panel to initialize the RAG Knowledge Matrix.")

# --- INTERACTIVE CHAT RUNTIME ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question regarding internal employee rules:", disabled=(retriever is None)):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Processing official guidelines..."):
            # Guardrails screening
            guard_prompt = ChatPromptTemplate.from_template(
                "Analyze scope. Respond only 'IN_SCOPE' or 'OUT_OF_SCOPE'. Consider Zyro and Acrux items in scope. Question: {question}"
            )
            guard_chain = guard_prompt | llm | StrOutputParser()
            scope = guard_chain.invoke({"question": user_query}).strip()
            
            if "OUT_OF_SCOPE" in scope:
                ans_text = REFUSAL_MESSAGE
                st.markdown(ans_text)
            else:
                retrieved_chunks = retriever.invoke(user_query)
                context_data = "\n\n".join(c.page_content for c in retrieved_chunks)
                
                rag_prompt = ChatPromptTemplate.from_template(
                    "You are the official HR Chatbot for Zyro Dynamics (Acrux Dynamics). Answer based exclusively on the context below. Context: {context}\nQuestion: {question}"
                )
                rag_chain_execution = (
                    {"context": lambda x: context_data, "question": RunnablePassthrough()}
                    | rag_prompt
                    | llm
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
