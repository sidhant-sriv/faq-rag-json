# Yantra FAQ API Documentation

This API provides endpoints for managing and querying FAQ data for the Yantra hackathon event.

## Endpoints

### Initialize Data
Initializes the vector store with FAQ data from the base dataset. Depends on data.json

- **URL**: `/initialize`
- **Method**: `POST`
- **Request Body**: None
- **Response**:
    ```json
    {
        "status": "Data initialized successfully"
    }
    ```
- **Error Response**:
    ```json
    {
        "error": "Error message"
    }
    ```

### Add Data
Adds new FAQ entries to the existing vector store.

- **URL**: `/add_data`
- **Method**: `POST`
- **Request Body**:
    ```json
    {
        "faqs": [
            {
                "question": "What is Yantra?",
                "answer": "Yantra is a hackathon event..."
            }
        ]
    }
    ```
- **Response**:
    ```json
    {
        "status": "Data added successfully"
    }
    ```
- **Error Response**:
    ```json
    {
        "error": "Validation error details"
    }
    ```

### Ask Question
Queries the FAQ system using an ensemble of BM25 and vector search retrievers.

- **URL**: `/ask`
- **Method**: `POST`
- **Request Body**:
    ```json
    {
        "question": "When is Yantra happening?"
    }
    ```
- **Response**:
    ```json
    {
        "result": "Answer to the question based on FAQ data"
    }
    ```
- **Error Response**:
    ```json
    {
        "error": "Error message"
    }
    ```

## Error Handling

All endpoints return a 500 status code for server errors and include an error message in the response body. The `/add_data` endpoint returns a 400 status code for validation errors, and the `/ask` endpoint returns a 400 status code when the question is missing from the request.

## Dependencies
- Flask
- Supabase
- LangChain
- Cohere
- Groq
- Pydantic

## Environment Variables
Create a `.env` file in the root directory with the following variables:
```
COHERE_API=your-cohere-api-key
SUPABASE_SERVICE_KEY=your-supabase-key
SUPABASE_URL=your-supabase-url
GROQ_API_KEY=your-groq-api-key
```

## Running with Docker

### Prerequisites
- Docker installed on your system

### Steps
1. Build the Docker image:

        
    `docker build -t yantra-faq-api .`
        

2. Run the container:
    `docker run -p 5002:5002 yantra-faq-api`
        

The API will be available at `http://localhost:5002`
