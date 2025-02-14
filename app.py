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
    logging.info(f"Selected COHERE API key index: {current_cohere_key_index}")
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
    logging.info(f"Selected GROQ API key index: {current_groq_key_index}")
    current_groq_key_index = (current_groq_key_index + 1) % len(GROQ_API_KEYS)
    return key


REDIS_URL = os.environ.get("REDIS_URL")
logging.info(f"REDIS_URL: {REDIS_URL}")
CACHE_EXPIRATION = int(os.environ.get("CACHE_EXPIRATION", 36000))
FUZZY_MATCH_THRESHOLD = float(os.environ.get("FUZZY_MATCH_THRESHOLD", 90.0))
RATE_LIMIT = os.environ.get("RATE_LIMIT", "100 per hour")

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
if SUPABASE_URL is None or SUPABASE_SERVICE_KEY is None:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment variables")

if not DISCORD_WEBHOOK_URL:
    logging.warning("DISCORD_WEBHOOK_URL is not set. Discord notifications will be skipped.")
else:
    logging.info(f"DISCORD_WEBHOOK_URL is set to: {DISCORD_WEBHOOK_URL}")

if REDIS_URL is None:
    raise ValueError("REDIS_URL must be set in the environment variables")
redis_client = redis.Redis.from_url(REDIS_URL)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    default_limits=[RATE_LIMIT],
)

data_cache = None


class FaqEntry(BaseModel):
    question: str
    answer: str


class FaqsModel(BaseModel):
    faqs: List[FaqEntry]


def get_cache_key(question: str) -> str:
    key = hashlib.md5(question.lower().strip().encode()).hexdigest()
    logging.debug(f"Generated cache key {key} for question: {question}")
    return key


def get_fuzzy_cache_match(question: str) -> tuple[dict, float]:
    logging.info(f"Searching for fuzzy cache match for: {question}")
    try:
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
            logging.debug(f"Comparing '{question}' with '{original_question}': ratio {ratio}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_question = original_question
                best_match_data = cached_data

        if best_ratio >= FUZZY_MATCH_THRESHOLD:
            logging.info(f"Found fuzzy match with ratio {best_ratio}% for question: {best_match_question}")
            return best_match_data, best_ratio

        logging.info("No sufficient fuzzy cache match found.")
        return {}, 0.0

    except Exception as e:
        logging.error(f"Error in fuzzy cache matching: {str(e)}")
        return {}, 0.0


def load_data():
    global data_cache
    if data_cache is None:
        logging.info("Loading data from data.json")
        loader = JSONLoader(file_path="data.json", text_content=False, jq_schema=".faqs[]")
        data_cache = loader.load()
        logging.info(f"Loaded {len(data_cache)} FAQ entries from file.")
    else:
        logging.info("Data already loaded; using cached data.")
    return data_cache


def send_to_discord_webhook(question: str) -> None:
    logging.info(f"Sending question to Discord webhook: {question}")
    if not DISCORD_WEBHOOK_URL:
        logging.warning("No DISCORD_WEBHOOK_URL is set. Skipping Discord notification.")
        return

    payload = {"content": f"An unanswered question was asked:\n**{question}**"}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, verify=False)
        if response.status_code != 204:
            logging.error(f"Failed to send question to Discord. Status: {response.status_code}, Response: {response.text}")
        else:
            logging.info("Question successfully posted to Discord.")
    except Exception as ex:
        logging.error(f"Exception while sending to Discord: {ex}")


@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": str(e)}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    logging.warning("Rate limit exceeded")
    return jsonify({"error": f"Rate limit exceeded: {e.description}"}), 429


@app.route("/health", methods=["GET"])
def health():
    logging.info("Health check endpoint called")
    return jsonify({"status": "ok"}), 200


@app.route("/add_data", methods=["POST"])
def add_data():
    logging.info("add_data endpoint called")
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
            logging.warning("Unauthorized access attempt in add_data endpoint")
            return jsonify({"error": "Unauthorized"}), 401

        new_json = request.get_json()
        if not new_json:
            logging.warning("No JSON payload provided in add_data endpoint")
            return jsonify({"error": "No JSON payload provided"}), 400

        faqs_model = FaqsModel(**new_json)
        logging.info(f"Received {len(faqs_model.faqs)} FAQ entries to add.")

        if not os.path.exists("data.json"):
            existing_data = {"faqs": []}
            logging.info("data.json does not exist. Creating new file.")
        else:
            with open("data.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            logging.info("Loaded existing data from data.json")

        existing_faqs = existing_data.get("faqs", [])
        for entry in faqs_model.faqs:
            existing_faqs.append({"question": entry.question, "answer": entry.answer})
            logging.debug(f"Appending FAQ: {entry.question}")

        with open("data.json", "w", encoding="utf-8") as f:
            json.dump({"faqs": existing_faqs}, f, indent=2, ensure_ascii=False)
        logging.info("Updated data.json with new FAQ entries.")

        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        items = []
        for entry in faqs_model.faqs:
            doc = Document(
                page_content=entry.answer,
                metadata={"question": entry.question},
            )
            items.extend(splitter.split_documents([doc]))
        logging.info(f"Split documents into {len(items)} chunks.")

        import asyncio
        asyncio.run(vector_store.aadd_documents(items))
        logging.info("Documents added to vector store.")

        redis_client.flushdb()
        logging.info("Cleared Redis cache.")

        return jsonify({"status": "Data added successfully"}), 200

    except ValidationError as ve:
        logging.error(f"Validation error: {ve.errors()}")
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        return handle_exception(e)


@app.route("/ask", methods=["POST"])
def ask():
    logging.info("ask endpoint called")
    json_data = request.get_json()
    if not json_data or "question" not in json_data:
        logging.warning("Question not provided in ask endpoint")
        return jsonify({"error": "Question is required"}), 400

    question = json_data["question"]
    logging.info(f"Received question: {question}")

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
    logging.info(f"Using GROQ API Key: {GROQ_API_KEY}")

    try:
        cached_response, match_ratio = get_fuzzy_cache_match(question)
        if cached_response:
            logging.info(f"[Cache Hit] Fuzzy match ratio: {match_ratio}%")
            cached_response["cache_hit"] = True
            cached_response["fuzzy_match_ratio"] = match_ratio
            return jsonify(cached_response)

        llm = ChatGroq(
            model="llama-3.3-70b-specdec",
            temperature=0.0,
            max_retries=2,
            api_key=SecretStr(GROQ_API_KEY),
        )

        template = """
        Question: {question}

        You are an FAQ assistant for the Yantra ’25 event. Your role is to provide clear, detailed, and accurate answers based only on the given information. Follow these guidelines when responding:
        
        Fact-Based Responses: Provide answers strictly based on the given information. Do not speculate or include any details not explicitly provided.
        
        Helpful Resources: When relevant, suggest official resources such as websites or social media pages to guide the user further.
        
        Acknowledging Unavailable Information: If the answer is not available, simply respond with: “I don’t know.”
        
        Focused and Complete Answers: Ensure responses are concise yet thorough, addressing the user’s query without unnecessary commentary.
        
        Neutral Tone: Maintain a neutral tone and do not reference or discuss the context, availability of information, or the source of your knowledge.
        
        Make sure to follow all instructions. Do not ignore at any cost. 

        Never ever comment on the context.

        {context}
        
        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        logging.info("Prompt template prepared for LLM.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )
        response = qa_chain({"query": question})
        logging.info(f"LLM Response: {response}")

        cache_response = {k: v for k, v in response.items() if k != "context"}
        cache_response["original_question"] = question
        cache_response["cache_hit"] = False

        if "I don’t know." in cache_response.get("result", ""):
            logging.info("LLM responded with 'I don’t know.' Sending notification to Discord.")
            send_to_discord_webhook(question)

        cache_key = f"qa:{get_cache_key(question)}"
        redis_client.setex(
            cache_key,
            timedelta(seconds=CACHE_EXPIRATION),
            json.dumps(cache_response),
        )
        logging.info(f"Stored response in cache with key: {cache_key}")

        if cache_response["result"] == "I don't know.":
            send_to_discord_webhook(cache_response["original_question"])

        return jsonify(cache_response)

    except Exception as e:
        return handle_exception(e)


if __name__ == "__main__":
    logging.info("Starting Flask server on port 5002")
    app.run(host="0.0.0.0", port=5002, debug=True)
