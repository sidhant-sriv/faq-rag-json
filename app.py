from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain import hub
import asyncio
import logging
from pydantic import BaseModel, ValidationError, SecretStr
from typing import List
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

COHERE_API = os.environ.get("COHERE_API")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if SUPABASE_URL is None or SUPABASE_SERVICE_KEY is None:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment variables"
    )

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Initialize Flask app
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Global variables for caching loaded data and embeddings
data_cache = None

# Pydantic models for request validation
class FaqEntry(BaseModel):
    question: str
    answer: str

class FaqsModel(BaseModel):
    faqs: List[FaqEntry]

embeddings = CohereEmbeddings(cohere_api_key=SecretStr(COHERE_API) if COHERE_API else None, model="embed-english-v3.0", client=None, async_client=None)
vector_store = SupabaseVectorStore(client=supabase, table_name="documents", query_name="match_documents", embedding=embeddings)

# Asynchronous data loader function with caching
async def load_data():
    global data_cache
    if data_cache is None:
        loader = JSONLoader(file_path="data.json", text_content=False, jq_schema=".faqs[]")
        data_cache = await asyncio.to_thread(loader.load)
    return data_cache

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

@app.route("/initialize", methods=["POST"])
async def initialize():
    logging.info("Initializing data...")
    try:
        data = await load_data()
        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        texts = splitter.split_documents(data)
        await vector_store.aadd_documents(texts)
        return jsonify({"status": "Data initialized successfully"})
    except Exception as e:
        return handle_exception(e)

@app.route("/add_data", methods=["POST"])
async def add_data():
    new_json = request.get_json()
    
    try:
        faqs_model = FaqsModel(**new_json)  # Validate incoming JSON
        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        items = splitter.split_documents(
            [
                Document(
                    page_content=entry.answer,
                    metadata={"question": entry.question},
                )
                for entry in faqs_model.faqs
            ]
        )

        await vector_store.aadd_documents(items)
        return jsonify({"status": "Data added successfully"})
    
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400

    except Exception as e:
        return handle_exception(e)

@app.route("/ask", methods=["POST"])
async def ask():
    json_data = request.get_json()
    
    if not json_data or "question" not in json_data:
        return jsonify({"error": "Question is required"}), 400
    
    question = json_data["question"]

    try:
        # Load and process data for retrieval
        data = await load_data()
        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        texts = splitter.split_documents(data)

        bm25_retriever = BM25Retriever.from_documents(texts)
        bm25_retriever.k = 2
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_store.as_retriever()], 
            weights=[0.5, 0.5]
        )

        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.0, max_retries=2)
        PROMPT = hub.pull("rlm/rag-prompt")

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            chain_type_kwargs={"prompt": PROMPT},
        )

        response = qa(
            {
                "query": question,
                "context": "You are an FAQ assistant for a hackathon event called Yantra. Only use relevant context and give relevant answers.",
            }
        )
        
        return jsonify(response)

    except Exception as e:
        return handle_exception(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002,debug=True)
