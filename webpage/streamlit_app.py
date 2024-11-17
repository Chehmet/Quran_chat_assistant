import streamlit as st
import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset and RAG model globally
@st.cache_resource
def load_resources():
    # Load the dataset
    try:
        qa_dataset = pd.read_csv("processed_quran_qa_data.csv")  # Update with the correct path
        st.write("Dataset loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None

    # Load the RAG pipeline
    try:
        pipeline = joblib.load("best_rag_pipeline.pkl")  # Ensure the file is accessible
        st.write("Pipeline loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None, None

    return qa_dataset, pipeline

qa_dataset, pipeline = load_resources()

# TF-IDF and similarity calculation function
def get_most_similar_question(query, qa_dataset, pipeline):
    if qa_dataset is None or pipeline is None:
        return {"advice": "Resources not initialized correctly.", "context": ""}

    # Ensure all questions are strings
    qa_dataset['question_en'] = qa_dataset['question_en'].astype(str)

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(qa_dataset['question_en'])

    # Transform the query
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors)

    # Get most similar question
    most_similar_idx = similarities.argmax()
    similarity_score = similarities[0, most_similar_idx]

    if similarity_score >= 0.7:  # Threshold can be adjusted
        answer_en = qa_dataset.iloc[most_similar_idx]['answer_en']
        context = qa_dataset.iloc[most_similar_idx]['context']
    else:
        st.warning("No sufficient match found. Using pipeline...")
        result = pipeline.run(query=query, params={"Reader": {"top_k": 1}})
        answer = result["answers"][0] if result["answers"] else None
        answer_en = answer.answer if answer else "No relevant answer found."
        context = answer.context if answer else ""

    return {"advice": answer_en, "context": context}

# Streamlit UI
st.title("🌙 Quran Answerer App")
st.subheader("Ask a question and get insights from the Quran Tafseer.")

question = st.text_input("Enter your question:")

if st.button("Get answer"):
    if not question:
        st.error("Please enter a question.")
    else:
        response = get_most_similar_question(question, qa_dataset, pipeline)
        st.write("### Short answer:")
        st.write(response["advice"])
        st.write("### Context:")
        st.write(response["context"])
