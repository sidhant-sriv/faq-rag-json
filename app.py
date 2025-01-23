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
import logging
from pydantic import BaseModel, ValidationError, SecretStr
from typing import List
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
FUZZY_MATCH_THRESHOLD = float(
    os.environ.get("FUZZY_MATCH_THRESHOLD", 90.0)
)  # Default 90%

# 1. Discord Webhook
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # Add your webhook URL

if SUPABASE_URL is None or SUPABASE_SERVICE_KEY is None:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment variables"
    )

# Initialize Redis client
redis_client = redis.Redis.from_url(REDIS_URL)

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
    """Generate a consistent cache key for a question."""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def get_fuzzy_cache_match(question: str) -> tuple[dict, float]:
    """
    Find the best fuzzy match for a question in the Redis cache.
    Returns (cached_response_dict, match_ratio) if found above threshold,
    otherwise ({}, 0.0).
    """
    try:
        # Safely iterate through Redis keys with SCAN to avoid large blocking
        # queries that KEYS can cause in production environments.
        all_keys = redis_client.scan_iter("qa:*")

        best_match_question = None
        best_match_data = {}
        best_ratio = 0.0

        for key in all_keys:
            raw_data = redis_client.get(key)
            if not raw_data:
                continue  # Skip if there's no data or the key was removed
            try:
                cached_data = json.loads((raw_data).decode("utf-8"))
            except json.JSONDecodeError:
                continue  # Skip if the data is invalid JSON

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
    import requests

    if not DISCORD_WEBHOOK_URL:
        logging.warning("No DISCORD_WEBHOOK_URL is set. Skipping Discord notification.")
        return

    payload = {"content": f"An unanswered question was asked:\n**{question}**"}

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        # Discord webhook returns status 204 on success
        if response.status_code != 204:
            logging.error(
                f"Failed to send question to Discord. Status: {response.status_code}, Response: {response.text}"
            )
    except Exception as ex:
        logging.error(f"Exception while sending to Discord: {ex}")


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


@app.route("/initialize", methods=["POST"])
def initialize():
    logging.info("Initializing data...")
    try:
        data = load_data()
        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        texts = splitter.split_documents(data)

        # Because vector_store.aadd_documents is async, we can run it in a blocking way:
        import asyncio

        asyncio.run(vector_store.aadd_documents(texts))

        return jsonify({"status": "Data initialized successfully"})
    except Exception as e:
        return handle_exception(e)


@app.route("/add_data", methods=["POST"])
def add_data():
    """
    Endpoint to add new FAQ data. Validates the structure and updates the
    local JSON file, the vector store, and clears the Redis cache
    to avoid stale data. Requires an API key in the headers.
    """
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
            # Split each new FAQ into smaller chunks if needed
            items.extend(splitter.split_documents([doc]))

        import asyncio

        asyncio.run(vector_store.aadd_documents(items))

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

    try:
        # Check cache first using fuzzy matching
        cached_response, match_ratio = get_fuzzy_cache_match(question)
        if cached_response:
            logging.info(f"[Cache Hit] Fuzzy match ratio: {match_ratio}%")
            cached_response["cache_hit"] = True
            cached_response["fuzzy_match_ratio"] = match_ratio

            # 3a. If the result is "I don't know" (from cache), notify Discord.
            if cached_response.get("result") == "I don't know":
                send_to_discord_webhook(question)
            return jsonify(cached_response)

        llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0.0, max_retries=2)

        template = """
        You are an FAQ assistant for the Yantra '25 event. Your primary goal is to provide clear, detailed, and accurate answers using ONLY the information provided in the given context. If relevant, you may also suggest additional resources like social media handles or official websites. 

        Follow these guidelines when responding:
        1. Base your answers strictly on the provided context. 
        2. If a user asks for more information or external resources, provide links or references as appropriate (e.g., social media pages, official sites).
        3. When you do not have enough information from the context, respond with "I don't know" rather than creating content or speculating.
        4. Do not invent answers or details outside of what the context provides.
        5. Strive to give comprehensive, concise responses that fully address the user's question.
        6. Do not mention anything about the context not knowing something.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "context",
                "question",
            ],
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )
        response = qa_chain({"query": question})  # response is typically a dict

        # Prepare response for caching
        # We exclude the "context" key because it can be large/unnecessary
        cache_response = {k: v for k, v in response.items() if k != "context"}
        cache_response["original_question"] = question
        cache_response["cache_hit"] = False

        # 3b. If the LLM result is "I don't know", send the question to the Discord webhook.
        if cache_response.get("result") == "I don't know":
            send_to_discord_webhook(question)

        # Store in Redis cache
        cache_key = f"qa:{get_cache_key(question)}"
        redis_client.setex(
            cache_key,
            timedelta(seconds=CACHE_EXPIRATION),
            json.dumps(cache_response),
        )

        return jsonify(cache_response)

    except Exception as e:
        return handle_exception(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
