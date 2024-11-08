# Install FastAPI and Uvicorn
# pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the RAG pipeline
pipeline = joblib.load("rag_pipeline.pkl")

class QueryRequest(BaseModel):
    question: str

@app.post("/get_advice")
async def get_advice(request: QueryRequest):
    query = request.question
    result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
    
    # Get the most relevant answer and its context
    advice = result["answers"][0].answer if result["answers"] else "No answer found."
    context = result["answers"][0].context if result["answers"] else ""
    return {"advice": advice, "context": context}

# Run the server with `uvicorn APIfile:app --reload`
