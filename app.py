import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()  # Load .env if present locally

app = FastAPI()

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
    answer = qa_chain.run(query.question)
    return {"question": query.question, "answer": answer}
