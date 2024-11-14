import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
import joblib

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
        content=f"Surah {row['Surah']} Ayat {row['Ayat']}: {row['Tafseer']} - {row['Description']}",
        meta={'type': 'tafseer', 'surah': row['Surah'], 'ayat': row['Ayat']}
    )
    for _, row in tafseer_dataset.iterrows()
]

# Write documents from QA dataset with metadata indicating it’s from the QA dataset
qa_documents = [
    Document(
        content=f"Question: {row['question_en']} - Answer: {row['answer_en']}",
        meta={'type': 'qa', 'question': row['question_en']}
    )
    for _, row in qa_dataset.iterrows()
]

# Write both document lists to the document store
document_store.write_documents(tafseer_documents + qa_documents)

# Initialize BM25 Retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize a more robust reader model
reader = TransformersReader(model_name_or_path="deepset/roberta-large-squad2", top_k=3)

# Set up the pipeline with the BM25 retriever and the large reader model
pipeline = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)

# Save the pipeline for reuse
joblib.dump(pipeline, "rag_pipeline_combined_optimized.pkl")
print("Optimized pipeline saved successfully.")

# Function to test the RAG pipeline with improved answer extraction and metadata handling
def test_rag(question):
    # Load the optimized pipeline
    pipeline = joblib.load("rag_pipeline_combined_optimized.pkl")
    
    # Run query with tuned parameters
    result = pipeline.run(
        query=question,
        params={
            "Retriever": {"top_k": 15},  # Retrieve more passages for better coverage
            "Reader": {"top_k": 5}       # Increase reader top_k to find better answers
        }
    )
    
    # Extract the best answer with additional metadata
    if result['answers']:
        best_answer = max(result['answers'], key=lambda x: x.score)  # Pick the highest confidence answer
        answer = best_answer.answer
        confidence = best_answer.score
        context = best_answer.context
        source_type = best_answer.meta.get('type', 'Unknown')
        surah = best_answer.meta.get('surah', 'Unknown') if source_type == 'tafseer' else None
        ayat = best_answer.meta.get('ayat', 'Unknown') if source_type == 'tafseer' else None
        
        # Prepare output with source-specific information
        if source_type == 'tafseer':
            explanation = f"In Surah {surah}, Ayat {ayat}, it is mentioned: {context}."
        else:
            explanation = f"The QA dataset provided the answer from context: {context}."
        
        output = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "context": context,
            "source_type": source_type,
            "surah": surah,
            "ayat": ayat,
            "explanation": explanation
        }
    else:
        output = {
            "question": question,
            "answer": "Sorry, I couldn't find an answer to your question.",
            "confidence": None,
            "context": None,
            "source_type": None,
            "surah": None,
            "ayat": None
        }
    
    return output

# Example usage
while True:
    question = input("Enter your question: ")
    result = test_rag(question)
    print(result)
