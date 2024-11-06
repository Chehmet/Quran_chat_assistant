# Install necessary packages
# !pip install haystack transformers

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
import pandas as pd


# Load processed data
data = pd.read_csv('processed_quran_data.csv')

# Initialize Document Store and add documents
document_store = InMemoryDocumentStore()
documents = [
    Document(
        content=f"Surah {row['Surah']} Ayat {row['Ayat']}: {row['Tafseer']}",
        meta={'surah': row['Surah'], 'ayat': row['Ayat']}
    )
    for _, row in data.iterrows()
]
document_store.write_documents(documents)

# Retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True
)
document_store.update_embeddings(retriever)

# Reader Model
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

# RAG Pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)
