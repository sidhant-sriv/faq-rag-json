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

# Load environment variables
load_dotenv()

from pydantic import SecretStr

COHERE_API = SecretStr(os.environ["COHERE_API"]) if os.environ.get("COHERE_API") else None
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
app = Flask(__name__)

# Asynchronous data loader function
async def load_data():
    loader = JSONLoader(file_path="sample.json", text_content=False, jq_schema=".faqs[]")
    return await asyncio.to_thread(loader.load)

@app.route('/ask', methods=['POST'])
async def ask():
    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'Invalid JSON data'}), 400
    question = json_data.get('question')
    if not question:
        return jsonify({'error': 'Question parameter is required'}), 400

    # Load and process data asynchronously
    data = await load_data()
    splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=300)
    texts = splitter.split_documents(data)

    # Initialize retrievers
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 2
    
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API, model="embed-english-v3.0", client=None, async_client=None)
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        embedding=embeddings
    )
    await vector_store.aadd_documents(texts)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store.as_retriever()], weights=[0.5, 0.5]
    )

    # Initialize LLM and QA chain
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.0,
        max_retries=2,
    )
    
    PROMPT = hub.pull("rlm/rag-prompt")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=ensemble_retriever, chain_type_kwargs={"prompt": PROMPT}
    )

    # Get response from QA chain asynchronously
    response =  qa({
        "query": question,
        "context": "You are an FAQ assistant. Do not give anything irrelevant. If it doesn't exist in the context, do not return it"
    })

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
    # Example curl command to test the /ask endpoint
    # Save this as a shell script or run it directly in the terminal

