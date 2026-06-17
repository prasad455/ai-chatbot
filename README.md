# 📄 PDF Chatbot using RAG

An AI-powered chatbot that answers questions from any PDF document.

## 🔧 Tech Stack
- LangChain
- Google Gemini API
- HuggingFace Embeddings
- ChromaDB (Vector Database)
- Streamlit

## 🚀 How it works
1. Upload any PDF
2. App splits it into chunks and stores in vector database
3. Ask any question
4. App finds relevant chunks and Gemini answers accurately

## ▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

## 💡 Use Cases
- Chat with study notes
- Query legal documents
- Understand research papers