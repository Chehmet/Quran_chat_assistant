# Install FastAPI and Uvicorn
# pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn import pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/get_advice")
async def get_advice(request: QueryRequest):
    query = request.question
    result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
    
    # Get the most relevant answer and its context
    advice = result["answers"][0].answer
    context = result["answers"][0].context
    return {"advice": advice, "context": context}

# Run the server with `uvicorn filename:app --reload`
