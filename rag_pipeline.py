from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
import pandas as pd

# Load processed data
data = pd.read_csv('processed_quran_data.csv')

# Ensure consistent quotation for all Tafseer entries
data['Tafseer'] = data['Tafseer'].apply(lambda x: f'"{x}"' if not x.startswith('"') else x)

# Initialize document store
document_store = InMemoryDocumentStore()

# Write documents with formatted content for clarity
documents = [
    Document(
        content=f"Surah {row['Surah']} Ayat {row['Ayat']}: {row['Tafseer']}",
        meta={'surah': row['Surah'], 'ayat': row['Ayat']}
    )
    for _, row in data.iterrows()
]
document_store.write_documents(documents)

# Define and configure retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True
)
document_store.update_embeddings(retriever)

# Define and configure reader with confidence threshold
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, top_k=1)

# Load into pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)

# Save pipeline for reuse
import joblib
joblib.dump(pipeline, "rag_pipeline.pkl")
