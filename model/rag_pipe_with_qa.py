import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Tafseer dataset
tafseer_dataset = pd.read_csv('datasets/processed_quran_data.csv')
tafseer_dataset['Description'] = tafseer_dataset.apply(lambda row: f"This ayat discusses: {row.get('Theme', 'No specific theme')}.", axis=1)

# Load QA dataset
qa_dataset = pd.read_csv('datasets/processed_quran_qa_data.csv')

# Initialize the document store
document_store = InMemoryDocumentStore(use_bm25=True)

# Write documents from Tafseer dataset with descriptive metadata
tafseer_documents = [
    Document(
        content=f"Surah {row['Surah']} Ayat {row['Ayat']}: {row['Tafseer']}",
        meta={'type': 'tafseer', 'surah': row['Surah'], 'ayat': row['Ayat']}
    )
    for _, row in tafseer_dataset.iterrows()
]

# Write documents from QA dataset with answer-only content and metadata
qa_documents = [
    Document(
        content=row['answer_en'],
        meta={'type': 'qa', 'question': row['question_en']}
    )
    for _, row in qa_dataset.iterrows()
]

# Write both document lists to the document store
document_store.write_documents(tafseer_documents + qa_documents)

# Initialize BM25 Retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize a robust reader model
reader = TransformersReader(model_name_or_path="deepset/roberta-large-squad2", top_k=5, max_seq_len=512)

# Set up the pipeline with the BM25 retriever and the large reader model
pipeline = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)

# Save the pipeline for reuse
joblib.dump(pipeline, "rag_pipeline_combined_optimized.pkl")
print("Optimized pipeline saved successfully.")

# Function to check question similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_most_similar_question(query):
    # Ensure all questions are strings and handle any non-string values
    qa_dataset['question_en'] = qa_dataset['question_en'].astype(str)  # Convert all entries to string
    
    # Check for NaN values and replace them with an empty string if necessary
    qa_dataset['question_en'] = qa_dataset['question_en'].fillna('')

    # Encode questions
    question_embeddings = similarity_model.encode(qa_dataset['question_en'].tolist())
    
    # Encode the query
    query_embedding = similarity_model.encode([query])[0]
    
    similarities = cosine_similarity([query_embedding], question_embeddings)
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[0][most_similar_idx]
    
    return most_similar_idx, similarity_score
