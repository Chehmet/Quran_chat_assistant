# rag_pipeline.py
import joblib
import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, BM25Retriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document

# Load the preprocessed dataset
qa_data = pd.read_csv("processed_quran_qa_data.csv")

# Initialize the DocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Convert each row to a Document with context data
documents = [
    Document(
        content=row['context_data'],  # Full context for retrieval
        meta={
            'question': row['question_en'],
            'answer': row['answer_en'],
            'context': row['context']
        }
    )
    for _, row in qa_data.iterrows()
]

# Write documents to the DocumentStore
document_store.write_documents(documents)


# Use BM25Retriever for basic retrieval
bm25_retriever = BM25Retriever(document_store=document_store)

# Use DensePassageRetriever for better context matching (optional)
dpr_retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
)
document_store.update_embeddings(dpr_retriever)

# Set up a reader with a pre-trained model
reader = TransformersReader(model_name_or_path="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

# Create pipelines for both retrievers
pipeline_bm25 = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)
pipeline_dpr = ExtractiveQAPipeline(reader=reader, retriever=dpr_retriever)

# Save both pipelines
joblib.dump(pipeline_bm25, "rag_pipeline_bm25.pkl")
joblib.dump(pipeline_dpr, "rag_pipeline_dpr.pkl")
print("Both pipelines saved successfully.")
