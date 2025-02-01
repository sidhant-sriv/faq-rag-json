from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
from pydantic import BaseModel, ValidationError, SecretStr
from typing import List
from langchain.schema import Document
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
import json
import redis
import hashlib

from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

COHERE_API_KEYS = os.environ.get("COHERE_API_KEYS", "").split(",")
current_cohere_key_index = 0

def get_cohere_api_key():
    global current_cohere_key_index
    key = COHERE_API_KEYS[current_cohere_key_index]
    current_cohere_key_index = (current_cohere_key_index + 1) % len(COHERE_API_KEYS)
    return key

COHERE_API = get_cohere_api_key()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")

# --- ROUND-ROBIN GROQ API KEY SETUP ---
GROQ_API_KEYS = os.environ.get("GROQ_API_KEYS", "").split(",")
current_groq_key_index = 0

def get_groq_api_key():
    global current_groq_key_index
    key = GROQ_API_KEYS[current_groq_key_index]
    current_groq_key_index = (current_groq_key_index + 1) % len(GROQ_API_KEYS)
    return key
# --------------------------------------

# Load the REDIS_URL environment variable
REDIS_URL = os.environ.get("REDIS_URL")
print(REDIS_URL)

# Initialize GPTCache with Redis and FAISS (used for other caching if needed)
onnx = Onnx()
data_manager = manager_factory(
    "redis,faiss",
    eviction_manager="redis",
    scalar_params={"url": REDIS_URL},
    vector_params={"dimension": onnx.dimension},
    eviction_params={
        "maxmemory": "100mb",
        "policy": "allkeys-lru",
        "ttl": 1
    }
)
cache = Cache()
cache.init(data_manager=data_manager)

# Initialize a separate Redis client for the exact question check
redis_client = redis.Redis.from_url(REDIS_URL)

def get_exact_cache_key(question: str) -> str:
    """Generate a unique key for the question using MD5 hashing."""
    normalized = question.strip().lower().encode("utf-8")
    return f"qa_exact:{hashlib.md5(normalized).hexdigest()}"

RATE_LIMIT = os.environ.get("RATE_LIMIT", "100 per hour")

# 1. Discord Webhook
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # Add your webhook URL

if SUPABASE_URL is None or SUPABASE_SERVICE_KEY is None:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment variables"
    )

if not DISCORD_WEBHOOK_URL:
    logging.warning("DISCORD_WEBHOOK_URL is not set. Discord notifications will be skipped.")
else:
    logging.info(f"DISCORD_WEBHOOK_URL is set to: {DISCORD_WEBHOOK_URL}")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Initialize Flask app and CORS
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT]
)

# Global variable for caching loaded data
data_cache = None

# Pydantic models for request validation
class FaqEntry(BaseModel):
    question: str
    answer: str

class FaqsModel(BaseModel):
    faqs: List[FaqEntry]

def load_data():
    """Load the JSON data from a local file once and cache it in memory."""
    global data_cache
    if data_cache is None:
        loader = JSONLoader(
            file_path="data.json", text_content=False, jq_schema=".faqs[]"
        )
        data_cache = loader.load()
    return data_cache

def send_to_discord_webhook(question: str) -> None:
    """
    Sends the unanswered question to a configured Discord webhook URL.
    """
    if not DISCORD_WEBHOOK_URL:
        logging.warning("No DISCORD_WEBHOOK_URL is set. Skipping Discord notification.")
        return

    payload = {"content": f"An unanswered question was asked:\n**{question}**"}
    try:
        # response = requests.post(DISCORD_WEBHOOK_URL, json=payload, verify=False)
        if response.status_code != 204:
            logging.error(
                f"Failed to send question to Discord. Status: {response.status_code}, Response: {response.text}"
            )
        else:
            logging.info("Question successfully posted to Discord.")
    except Exception as ex:
        logging.error(f"Exception while sending to Discord: {ex}")

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

@app.route("/initialize", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def initialize():
    logging.info("Initializing data...")
    try:
        data = load_data()
        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        texts = splitter.split_documents(data)

        # Initialize vector store for adding documents
        COHERE_API = get_cohere_api_key()
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
        import asyncio
        asyncio.run(vector_store.aadd_documents(texts))
        return jsonify({"status": "Data initialized successfully"})
    except Exception as e:
        return handle_exception(e)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": f"Rate limit exceeded: {e.description}"
    }), 429

@app.route("/add_data", methods=["POST"])
def add_data():
    """
    Endpoint to add new FAQ data. Validates the structure and updates the local JSON file
    and the vector store. Requires an API key in the headers.
    """
    COHERE_API = get_cohere_api_key()
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
    try:
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != os.environ.get("API_KEY"):
            return jsonify({"error": "Unauthorized"}), 401

        new_json = request.get_json()
        if not new_json:
            return jsonify({"error": "No JSON payload provided"}), 400

        faqs_model = FaqsModel(**new_json)  # Expects {"faqs": [{question, answer}, ...]}

        if not os.path.exists("data.json"):
            existing_data = {"faqs": []}
        else:
            with open("data.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        existing_faqs = existing_data.get("faqs", [])
        for entry in faqs_model.faqs:
            existing_faqs.append({"question": entry.question, "answer": entry.answer})
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump({"faqs": existing_faqs}, f, indent=2, ensure_ascii=False)

        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        items = []
        for entry in faqs_model.faqs:
            doc = Document(
                page_content=entry.answer,
                metadata={"question": entry.question},
            )
            items.extend(splitter.split_documents([doc]))
        import asyncio
        asyncio.run(vector_store.aadd_documents(items))
        return jsonify({"status": "Data added successfully"}), 200

    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        return handle_exception(e)

@app.route("/ask", methods=["POST"])
def ask():
    """
    Main Q&A endpoint. Uses an ensemble of BM25 and Supabase VectorStore,
    then an LLM to generate an answer. First, it checks for an exact match of
    the question in Redis. If found, it returns the cached answer. Otherwise,
    it executes the query and then caches the result.
    """
    json_data = request.get_json()
    if not json_data or "question" not in json_data:
        return jsonify({"error": "Question is required"}), 400

    question = json_data["question"]

    # Check for an exact match in Redis
    exact_key = get_exact_cache_key(question)
    cached_exact = redis_client.get(exact_key)
    if cached_exact:
        logging.info("Exact cache hit for question")
        return jsonify({"result": cached_exact.decode("utf-8"), "cache_hit": True})

    COHERE_API = get_cohere_api_key()
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
    GROQ_API_KEY = get_groq_api_key()
    logging.log(msg=GROQ_API_KEY, level=1)
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-specdec",
            temperature=0.0,
            max_retries=2,
            api_key=SecretStr(GROQ_API_KEY)
        )

        template = """
            You are an FAQ assistant for the Yantra ’25 event. Your role is to provide clear, detailed, and accurate answers based only on the given information. Follow these guidelines when responding:
            1.  Fact-Based Responses: Provide answers strictly based on the given information. Do not speculate or include any details not explicitly provided.
            2.  Helpful Resources: When relevant, suggest official resources such as websites or social media pages to guide the user further.
            3.  Acknowledging Unavailable Information: If the answer is not available, simply respond with: “I don’t know.”
            4.  Focused and Complete Answers: Ensure responses are concise yet thorough, addressing the user’s query without unnecessary commentary.
            5.  Neutral Tone: Do not reference or discuss the availability of information or the source of your knowledge.

            NEVER COMMENT ON THE CONTEXT


            {context}

            Question:
            {question}

            Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )
        response = qa_chain({"query": question})
        result_text = response.get("result", "")

        # Cache the answer in Redis for future exact matches (TTL set to 1 hour)
        redis_client.setex(exact_key, 3600, result_text)

        # Send to Discord if the answer indicates missing information
        if "I don’t know." in result_text or "I don't know." in result_text:
            send_to_discord_webhook(question)

        return jsonify(response)

    except Exception as e:
        return handle_exception(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)