import os
import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv('GROQ_API_KEY')
if not api_key:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()
st.set_page_config(page_title="📘 PDF Q&A Chatbot", layout="wide")
st.title("📘 PDF Q&A Chatbot")

upload_file =st.file_uploader("upload pdf only",type=['pdf'])

if upload_file:
    with open("temp.pdf", "wb") as f:
        f.write(upload_file.read())
    st.info("📖 Reading your PDF...")
    progress = st.progress(0)

    # Step 1: Load PDF
    progress.progress(20)
    document = SimpleDirectoryReader(input_files=['temp.pdf']).load_data()

    # Step 2: Create embeddings
    progress.progress(60)
    emabading = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')

    # Step 3: Build index
    progress.progress(90)
    index=VectorStoreIndex.from_documents(document,embed_model=emabading)
    # Step 4: Finish
    progress.progress(100)
    st.success("✅ PDF processed successfully!")
    
    llm=Groq(model='llama-3.1-8b-instant',api_key=api_key)
    query_engin=index.as_query_engine(llm=llm)

    st.subheader("💬 Ask questions about your PDF")
    user_question = st.text_input("type your question here..")
    if user_question:
        response =query_engin.query(user_question)
        st.write(response.response)
