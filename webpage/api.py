import os
import gdown
import pickle
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

import joblib

try:
    pipeline = joblib.load("rag_pipeline_big_model.pkl")
    print("File loaded successfully.")
except Exception as e:
    print(f"Failed to load file with joblib: {e}")
    exit(0)


class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Quran Advice API. Use /get_advice endpoint to get advice."}

@app.post("/get_advice")
async def get_advice(request: QueryRequest):
    query = request.question
    if not hasattr(pipeline, "run"):
        raise AttributeError("The loaded pipeline does not have a 'run' method. Check the file or model loading.")
    result = pipeline.run(query=query, params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 1}})
    
    # Filter answer by confidence
    answer = result["answers"][0] if result["answers"] else None
    if answer and answer.score > 0.5:  # Confidence threshold (adjust as needed)
        advice = answer.answer
        context = answer.context
    else:
        advice = "No relevant answer found based on the Quran Tafseer."
        context = ""
    
    return {"advice": advice, "context": context}
