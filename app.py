import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(chunks, embedding=embeddings)
    return vector_store

def get_answer(vector_store, question):
    # Get relevant chunks using LangChain
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use Gemini directly for generation
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    You are a helpful assistant. Answer the question based only on the context below.
    If the answer is not in the context, say "Answer not found in the PDF."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.write("Upload a PDF and ask any question from it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        raw_text = get_pdf_text(uploaded_file)
        chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(chunks)
    st.success("✅ PDF processed! Ask your question below.")

    question = st.text_input("Ask a question from your PDF:")

    if question:
        with st.spinner("Finding answer..."):
            answer = get_answer(vector_store, question)
        st.write("### Answer:")
        st.write(answer)