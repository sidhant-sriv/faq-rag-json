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

if SUPABASE_URL is None or SUPABASE_SERVICE_KEY is None:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment variables"
    )

# Initialize Redis client
redis_client = redis.from_url(REDIS_URL)

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


async def get_fuzzy_cache_match(question: str) -> tuple[str, float]:
    """
    Find the best fuzzy match for a question in the cache.
    Returns the cached response and the match ratio if found, otherwise (None, 0).
    """
    try:
        # Get all cache keys
        all_keys = redis_client.keys("qa:*")
        if not all_keys:
            return None, 0

        # Extract original questions from cache
        cached_questions = []
        key_mapping = {}

        for key in all_keys:
            cached_data = json.loads(redis_client.get(key))
            original_question = cached_data.get("original_question", "")
            cached_questions.append(original_question)
            key_mapping[original_question] = key

        # Find best fuzzy match
        best_match = None
        best_ratio = 0

        for cached_question in cached_questions:
            ratio = fuzz.ratio(question.lower(), cached_question.lower())
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cached_question

        if best_ratio >= FUZZY_MATCH_THRESHOLD:
            cached_response = json.loads(redis_client.get(key_mapping[best_match]))
            return cached_response, best_ratio

        return None, 0

    except Exception as e:
        logging.error(f"Error in fuzzy cache matching: {str(e)}")
        return None, 0


# Asynchronous data loader function with caching
async def load_data():
    global data_cache
    if data_cache is None:
        loader = JSONLoader(
            file_path="data.json", text_content=False, jq_schema=".faqs[]"
        )
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
        # Clear Redis cache when new data is added
        redis_client.flushdb()
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
        # Check cache first using fuzzy matching
        cached_response, match_ratio = await get_fuzzy_cache_match(question)
        if cached_response:
            logging.info(f"Cache hit with fuzzy match ratio: {match_ratio}%")
            cached_response["cache_hit"] = True
            cached_response["fuzzy_match_ratio"] = match_ratio
            return jsonify(cached_response)

        # If no cache hit, proceed with the normal flow
        data = await load_data()
        splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=500)
        texts = splitter.split_documents(data)

        bm25_retriever = BM25Retriever.from_documents(texts)
        bm25_retriever.k = 3

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_store.as_retriever()], weights=[0.5, 0.5]
        )

        llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0.0, max_retries=2)

        template = """
        You are an FAQ assistant for the Yantra '25 event. Your primary goal is to provide clear, detailed, and accurate answers using ONLY the information provided in the given context. If relevant, you may also suggest additional resources like social media handles or official websites. 

        Follow these guidelines when responding:
        1. Base your answers strictly on the provided context. 
        2. If a user asks for more information or external resources, provide links or references as appropriate (e.g., social media pages, official sites).
        3. When you do not have enough information from the context, respond with "I don't know" rather than creating content or speculating.
        4. Do not invent answers or details outside of what the context provides.
        5. Strive to give comprehensive, concise responses that fully address the user's question.

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

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            chain_type_kwargs={"prompt": prompt},
        )
        response = qa(
            {
                "query": question,
            }
        )

        # Prepare response for caching
        cache_response = {k: v for k, v in response.items() if k != "context"}
        cache_response["original_question"] = question
        cache_response["cache_hit"] = False

        # Store in Redis cache
        cache_key = f"qa:{get_cache_key(question)}"
        redis_client.setex(
            cache_key, timedelta(seconds=CACHE_EXPIRATION), json.dumps(cache_response)
        )

        return jsonify(cache_response)

    except Exception as e:
        return handle_exception(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
