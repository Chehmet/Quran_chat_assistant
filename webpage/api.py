import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import uvicorn


app = FastAPI(title="Quran QA API", description="API for answering questions based on Quran data.", version="1.0")

# downloading weights from google drive
def download_model():
    file_id = "11jZu4D53kS6ny1ZhhwcX-HMa5HwxyDr2"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "best_rag_pipeline.pkl", quiet=False)

try:
    qa_dataset = pd.read_csv("processed_quran_qa_data.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(0)

try:
    download_model()
    pipeline = joblib.load("best_rag_pipeline.pkl")
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load pipeline: {e}")
    exit(0)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Quran Advice API. Use /get_advice endpoint to get advice."}

@app.post("/get_advice")
async def get_advice(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not loaded.")
    
    query = request.question
    response = get_most_similar_question(query)
    return response

def get_most_similar_question(query):
    qa_dataset['question_en'] = qa_dataset['question_en'].astype(str)
    vectorizer = TfidfVectorizer()

    question_vectors = vectorizer.fit_transform(qa_dataset['question_en'])
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, question_vectors)
    most_similar_idx = similarities.argmax()
    similarity_score = similarities[0, most_similar_idx]

    if similarity_score >= 0.7:
        answer_en = qa_dataset.iloc[most_similar_idx]['answer_en']
        context = qa_dataset.iloc[most_similar_idx]['context']
    else:
        print("No sufficient match found, using the pipeline...", similarity_score)
        result = pipeline.run(query=query, params={"Reader": {"top_k": 1}}) 
        answer = result["answers"][0] if result["answers"] else None
        answer_en = answer.answer if answer else "No relevant answer found."
        context = answer.context if answer else ""
    
    return {"advice": answer_en, "context": context}

if __name__ == "__main__":    
    uvicorn.run(app, host="127.0.0.1", port=8000)