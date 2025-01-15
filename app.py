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
from langchain.prompts import PromptTemplate
import asyncio
import logging
from pydantic import BaseModel, ValidationError, SecretStr
from typing import List, Tuple, Optional, Union
from langchain.schema import Document
import redis
from rapidfuzz import fuzz
import json
import hashlib
from datetime import timedelta

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

COHERE_API = os.environ.get("COHERE_API")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CACHE_EXPIRATION = int(os.environ.get("CACHE_EXPIRATION", 3600))  # Default 1 hour
FUZZY_MATCH_THRESHOLD = float(os.environ.get("FUZZY_MATCH_THRESHOLD", 90.0))  # Default 90%

if SUPABASE_URL is None or SUPABASE_SERVICE_KEY is None:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment variables"
    )

# Initialize Redis client
redis_client = redis.from_url(REDIS_URL, decode_responses=False)  
# decode_responses=False ensures we manually handle bytes <-> str

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

embeddings = CohereEmbeddings(
    cohere_api_key=SecretStr(COHERE_API) if COHERE_API else None,
    model="embed-english-v3.0",
    client=None,
    async_client=None,
)
vector_store = SupabaseVectorStore(
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    embedding=embeddings,
)

def get_cache_key(question: str) -> str:
    """
    Generate a consistent cache key for a given question.
    Using MD5 to ensure a short, unique string.
    """
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

async def get_fuzzy_cache_match(question: str) -> Tuple[Optional[dict], float]:
    """
    Find the best fuzzy match for a question in the cache.
    Returns (cached_response_dict, match_ratio) if found, otherwise (None, 0).
    """
    try:
        # Asynchronously retrieve Redis keys using scan_iter to avoid blocking
        # We'll accumulate them in a list for fuzzy matching
        all_keys = []
        # SCAN iterates in chunks, you need to loop until no more keys are returned
        async def _scan_redis(pattern: str):
            cursor = '0'
            while True:
                # Use to_thread to run blocking calls in a thread
                scan_result = await asyncio.to_thread(
                    redis_client.scan, cursor=int(cursor), match=pattern, count=100
                )
                cursor, keys = str(scan_result[0]), scan_result[1]
                all_keys.extend(keys)
                if cursor == '0':
                    break

        await _scan_redis('qa:*')

        if not all_keys:
            return None, 0

        cached_questions = []
        key_mapping = {}

        for key in all_keys:
            # key might be in bytes, ensure it's string
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            val_bytes = await asyncio.to_thread(redis_client.get, key_str)
            if val_bytes is None:
                continue  # Key has no value or doesn't exist
            val_str = val_bytes.decode('utf-8') if isinstance(val_bytes, bytes) else str(val_bytes)
            try:
                cached_data = json.loads(val_str)
            except json.JSONDecodeError:
                continue  # Skip if corrupted data

            original_question = cached_data.get('original_question', '').strip()
            if original_question:
                cached_questions.append(original_question)
                key_mapping[original_question] = key_str

        # Fuzzy match logic
        best_match = None
        best_ratio = 0.0
        lower_question = question.lower()
        for cached_question in cached_questions:
            ratio = fuzz.ratio(lower_question, cached_question.lower())
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cached_question

        if best_match and best_ratio >= FUZZY_MATCH_THRESHOLD:
            best_key = key_mapping[best_match]
            val_bytes = await asyncio.to_thread(redis_client.get, best_key)
            if val_bytes is None:
                return None, 0
            val_str = val_bytes.decode('utf-8') if isinstance(val_bytes, bytes) else val_bytes
            try:
                cached_response = json.loads(str(val_str))
            except json.JSONDecodeError:
                return None, 0
            return cached_response, best_ratio

        return None, 0

    except Exception as e:
        logging.error(f"Error in fuzzy cache matching: {str(e)}")
        return None, 0

# Asynchronous data loader function with caching
async def load_data():
    global data_cache
    if data_cache is None:
        loader = JSONLoader(file_path="data.json", text_content=False, jq_schema=".faqs[]")
        data_cache = await asyncio.to_thread(loader.load)
    return data_cache

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Exception occurred:")
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

        # Store in vector DB
        await vector_store.aadd_documents(items)

        # Clear Redis cache when new data is added to avoid stale data
        await asyncio.to_thread(redis_client.flushdb)
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

    question = json_data["question"].strip()

    try:
        # Check cache first using fuzzy matching
        cached_response, match_ratio = await get_fuzzy_cache_match(question)
        if cached_response:
            logging.info(f"[Cache Hit] Fuzzy match ratio: {match_ratio}% for question: '{question}'")
            cached_response["cache_hit"] = True
            cached_response["fuzzy_match_ratio"] = match_ratio
            return jsonify(cached_response)

        # If no cache hit, proceed with the normal retrieval flow
        data = await load_data()
        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        texts = splitter.split_documents(data)

        bm25_retriever = BM25Retriever.from_documents(texts)
        bm25_retriever.k = 3

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_store.as_retriever()],
            weights=[0.5, 0.5]
        )

        llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0.0, max_retries=2)

        template = """
        You are an FAQ assistant for Yantra '25 event. 

        Your job is to help with providing information about whatever is present in the given information.
        You should also suggest providing other sources of information like social media handles and websites when they are asked. 
        You are a helpful assistant that answers questions based on context in a clear manner.
        If you don't know the answer, just say "I don't know".
        Don't try to make up an answer.
        {context}

        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            chain_type_kwargs={"prompt": prompt},
        )

        response = qa({"query": question})

        # Strip out the context from the response before returning/caching
        cache_response = {k: v for k, v in response.items() if k != "context"}
        cache_response["original_question"] = question
        cache_response["cache_hit"] = False

        # Store in Redis cache with a generated key
        cache_key = f"qa:{get_cache_key(question)}"
        cache_value = json.dumps(cache_response)

        await asyncio.to_thread(
            redis_client.setex,
            cache_key,
            CACHE_EXPIRATION,  # seconds
            cache_value
        )

        return jsonify(cache_response)

    except Exception as e:
        return handle_exception(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)