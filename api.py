import os
import gdown
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pickle


app = FastAPI()

try:
    with open("rag_pipeline.pkl", "rb") as f:
        print(f.read(4)) 
        pipeline = pickle.load(f)
    print("File loaded successfully.")
except Exception as e:
    print(f"Failed to load file: {e}")
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
    result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
    advice = result["answers"][0].answer if result["answers"] else "No answer found."
    context = result["answers"][0].context if result["answers"] else ""
    return {"advice": advice, "context": context}
