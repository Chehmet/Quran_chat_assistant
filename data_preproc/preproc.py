from datasets import load_dataset
import pandas as pd

ds = load_dataset("M-AI-C/quran_tafseer", split="train")
df = ds.to_pandas()[['sorah', 'ayah', 'en-sarwar']]
df.columns = ['Surah', 'Ayat', 'Tafseer']

df.to_csv('processed_quran_data.csv', index=False)


qa_data = load_dataset("nazimali/quran-question-answer-context", split="train")
qa_data_df = qa_data.to_pandas()[['answer_en', 'question_en', 'context', 'context_data']]

qa_data_df = qa_data_df.dropna(subset=['answer_en', 'question_en', 'context', 'context_data'])
qa_data_df.to_csv("processed_quran_qa_data.csv", index=False)