import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load .env if present locally

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gray-mud-0ceefc21e.6.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Setup embedding and FAISS index ---
texts = [
    "Render is a great platform to deploy web apps easily.",
    "Retrieval-Augmented Generation (RAG) combines retrieval with generation.",
    "FAISS is used to perform efficient similarity search."
]

docs = [Document(page_content=chunk) for chunk in texts]

# Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Split & embed
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# FAISS vector store
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
    retriever=vectorstore.as_retriever()
)

# --- API Endpoint ---
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    answer = qa_chain.invoke(query.question)
    return {"question": query.question, "answer": answer}
