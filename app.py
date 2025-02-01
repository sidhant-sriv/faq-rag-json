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
import redis
from rapidfuzz import fuzz
import json
import hashlib
from datetime import timedelta
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests

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

REDIS_URL = os.environ.get("REDIS_URL")
print(REDIS_URL)
CACHE_EXPIRATION = int(os.environ.get("CACHE_EXPIRATION", 36000))  
FUZZY_MATCH_THRESHOLD = float(
    os.environ.get("FUZZY_MATCH_THRESHOLD", 90.0)
)  # Default 90%
RATE_LIMIT = os.environ.get("RATE_LIMIT", "100 per hour")  # New rate limit variable

# 1. Discord Webhook
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # Add your webhook URL

if SUPABASE_URL is None or SUPABASE_SERVICE_KEY is None:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment variables"
    )

# Log the status of DISCORD_WEBHOOK_URL
if not DISCORD_WEBHOOK_URL:
    logging.warning("DISCORD_WEBHOOK_URL is not set. Discord notifications will be skipped.")
else:
    logging.info(f"DISCORD_WEBHOOK_URL is set to: {DISCORD_WEBHOOK_URL}")

# Initialize Redis client
if REDIS_URL is None:
    raise ValueError("REDIS_URL must be set in the environment variables")
redis_client = redis.Redis.from_url(REDIS_URL)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Initialize Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    default_limits=[RATE_LIMIT]
)

# Global variables for caching loaded data and embeddings
data_cache = None

# Pydantic models for request validation
class FaqEntry(BaseModel):
    question: str
    answer: str

class FaqsModel(BaseModel):
    faqs: List[FaqEntry]


def get_cache_key(question: str) -> str:
    """Generate a consistent cache key for a question."""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def get_fuzzy_cache_match(question: str) -> tuple[dict, float]:
    """
    Find the best fuzzy match for a question in the Redis cache.
    Returns (cached_response_dict, match_ratio) if found above threshold,
    otherwise ({}, 0.0).
    """
    try:
        # Safely iterate through Redis keys with SCAN
        all_keys = redis_client.scan_iter("qa:*")

        best_match_question = None
        best_match_data = {}
        best_ratio = 0.0

        for key in all_keys:
            raw_data = redis_client.get(key)
            if not raw_data:
                continue
            try:
                cached_data = json.loads(raw_data.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            original_question = cached_data.get("original_question", "")
            if not original_question:
                continue

            ratio = fuzz.ratio(question.lower(), original_question.lower())
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_question = original_question
                best_match_data = cached_data

        if best_ratio >= FUZZY_MATCH_THRESHOLD:
            return best_match_data, best_ratio

        return {}, 0.0

    except Exception as e:
        logging.error(f"Error in fuzzy cache matching: {str(e)}")
        return {}, 0.0

def load_data():
    """Load the JSON data from local file once and cache it in memory."""
    global data_cache
    if data_cache is None:
        loader = JSONLoader(
            file_path="data.json", text_content=False, jq_schema=".faqs[]"
        )
        data_cache = loader.load()
    return data_cache

# 2. Define a function that sends questions to the Discord Webhook
def send_to_discord_webhook(question: str) -> None:
    """
    Sends the unanswered question to a configured Discord webhook URL.
    """

    if not DISCORD_WEBHOOK_URL:
        logging.warning("No DISCORD_WEBHOOK_URL is set. Skipping Discord notification.")
        return

    payload = {"content": f"An unanswered question was asked:\n**{question}**"}

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, verify=False)
        # Discord webhook returns status 204 on success
        if response.status_code != 204:
            logging.error(
                f"Failed to send question to Discord. "
                f"Status: {response.status_code}, Response: {response.text}"
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
    Endpoint to add new FAQ data. Validates the structure and updates the
    local JSON file, the vector store, and clears the Redis cache
    to avoid stale data. Requires an API key in the headers.
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
        # Check for API key in headers
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != os.environ.get("API_KEY"):
            return jsonify({"error": "Unauthorized"}), 401

        # Get the new FAQs from the request
        new_json = request.get_json()
        if not new_json:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Validate incoming JSON structure with Pydantic
        faqs_model = FaqsModel(**new_json)  # Expects {"faqs": [{question, answer}, ...]}

        if not os.path.exists("data.json"):
            # If data.json doesn't exist, create a basic structure
            existing_data = {"faqs": []}
        else:
            with open("data.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)

        existing_faqs = existing_data.get("faqs", [])

        # Append new FAQs
        for entry in faqs_model.faqs:
            existing_faqs.append({"question": entry.question, "answer": entry.answer})

        # Write the combined FAQs back to data.json
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

        # Clear Redis cache to avoid stale data
        redis_client.flushdb()

        return jsonify({"status": "Data added successfully"}), 200

    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        return handle_exception(e)

@app.route("/ask", methods=["POST"])
def ask():
    """
    Main Q&A endpoint. Checks Redis for a fuzzy cache match first.
    If not found, uses an ensemble of BM25 and Supabase VectorStore,
    then LLM to generate an answer. Caches the result in Redis.
    """
    json_data = request.get_json()
    if not json_data or "question" not in json_data:
        return jsonify({"error": "Question is required"}), 400

    question = json_data["question"]
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
    GROQ_API_KEY = get_groq_api_key()  # <<< Round-robin key
    print(GROQ_API_KEY)
    logging.log(msg=GROQ_API_KEY, level=1)
    try:
        # 1. Check cache first using fuzzy matching
        cached_response, match_ratio = get_fuzzy_cache_match(question)
        if cached_response:
            logging.info(f"[Cache Hit] Fuzzy match ratio: {match_ratio}%")
            cached_response["cache_hit"] = True
            cached_response["fuzzy_match_ratio"] = match_ratio
            return jsonify(cached_response)

        # 2. No cache match, query the LLM (using round-robin GROQ key)
        
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

            PS: The yantra website is https://www.yantra.swvit.in/ so keep that in mind regardless of context

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

        # Log the raw LLM response for debugging
        logging.info(f"LLM Response: {response}")

        # 3. Prepare response for caching
        cache_response = {k: v for k, v in response.items() if k != "context"}
        cache_response["original_question"] = question
        cache_response["cache_hit"] = False

        # 4. If the LLM result is "I don’t know", send the question to Discord (partial match).
        if "I don’t know." in cache_response.get("result", ""):
            send_to_discord_webhook(question)

        # 5. Store in Redis cache
        cache_key = f"qa:{get_cache_key(question)}"
        redis_client.setex(
            cache_key,
            timedelta(seconds=CACHE_EXPIRATION),
            json.dumps(cache_response),
        )

        # Also call Discord webhook if the final result is "I don't know." 
        if cache_response['result'] == "I don't know.":
            send_to_discord_webhook(cache_response['original_question'])

        return jsonify(cache_response)

    except Exception as e:
        return handle_exception(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
