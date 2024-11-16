import os
import pickle
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown

# FastAPI Setup
app = FastAPI()

# Function to download the .pkl file from Google Drive
def download_model():
    # Google Drive file ID from your link
    file_id = "11jZu4D53kS6ny1ZhhwcX-HMa5HwxyDr2"
    # Constructing the direct download URL
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "best_rag_pipeline.pkl", quiet=False)

# Load the dataset
try:
    qa_dataset = pd.read_csv("processed_quran_qa_data.csv")  # Adjust the path to your dataset
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(0)

# Download and load the RAG pipeline model
try:
    download_model()  # Download the model from Google Drive
    pipeline = joblib.load("best_rag_pipeline.pkl")
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load pipeline: {e}")
    exit(0)

# Define the QueryRequest model for FastAPI
class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Quran Advice API. Use /get_advice endpoint to get advice."}

@app.post("/get_advice")
async def get_advice(request: QueryRequest):
    query = request.question
    # Get the most similar question from the dataset
    response = get_most_similar_question(query)
    return response

def get_most_similar_question(query):
    # Ensure that all questions in the dataset are strings
    qa_dataset['question_en'] = qa_dataset['question_en'].astype(str)

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the dataset questions
    question_vectors = vectorizer.fit_transform(qa_dataset['question_en'])

    # Transform the query into the same vector space
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between the query and all questions
    similarities = cosine_similarity(query_vector, question_vectors)

    # Get the index of the most similar question
    most_similar_idx = similarities.argmax()
    similarity_score = similarities[0, most_similar_idx]

    # Define a threshold for similarity to consider a match (e.g., 0.7)
    if similarity_score >= 0.7:  # You can adjust this threshold
        answer_en = qa_dataset.iloc[most_similar_idx]['answer_en']  # Get the answer from the dataset
        context = qa_dataset.iloc[most_similar_idx]['context']  # Get the context from the dataset
    else:
        # Fallback to the pipeline if no good match is found
        print("No sufficient match found, using the pipeline...", similarity_score)
        result = pipeline.run(query=query, params={"Reader": {"top_k": 1}})  # Use pipeline for answers

        # Extract the answer and context from the pipeline response
        answer = result["answers"][0] if result["answers"] else None
        answer_en = answer.answer if answer else "No relevant answer found."
        context = answer.context if answer else ""
    
    return {"advice": answer_en, "context": context}

# Example usage:
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)