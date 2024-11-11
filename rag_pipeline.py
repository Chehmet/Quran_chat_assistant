from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.schema import Document
import pandas as pd
import joblib

# Load processed data
data = pd.read_csv('processed_quran_data.csv')

# Ensure consistent quotation for all Tafseer entries
data['Tafseer'] = data['Tafseer'].apply(lambda x: f'"{x}"' if not x.startswith('"') else x)

# Initialize document store with use_bm25=True
document_store = InMemoryDocumentStore(use_bm25=True)

# Write documents with formatted content for clarity
documents = [
    Document(
        content=f"Surah {row['Surah']} Ayat {row['Ayat']}: {row['Tafseer']}",
        meta={'surah': row['Surah'], 'ayat': row['Ayat']}
    )
    for _, row in data.iterrows()
]
document_store.write_documents(documents)

# Initialize BM25 retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize Dense Passage Retriever (DPR)
dpr_retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
)
document_store.update_embeddings(dpr_retriever)

# Initialize reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", top_k=1)

# Create pipeline and add components
pipeline = Pipeline()
pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
pipeline.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["BM25Retriever"])
pipeline.add_node(component=reader, name="Reader", inputs=["DPRRetriever"])

# Save pipeline for reuse
joblib.dump(pipeline, "rag_pipeline2.pkl")
print("Pipeline saved successfully.")
