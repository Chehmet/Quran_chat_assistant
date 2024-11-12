from datasets import load_dataset
import pandas as pd

# Load the dataset and specify the split
ds = load_dataset("M-AI-C/quran_tafseer", split="train")

# Convert to a DataFrame and select specific columns
df = ds.to_pandas()[['sorah', 'ayah', 'en-sarwar']]
df.columns = ['Surah', 'Ayat', 'Tafseer']

# Save the processed data for model input
df.to_csv('processed_quran_data.csv', index=False)


qa_data = load_dataset("nazimali/quran-question-answer-context", split="train")

# Convert to DataFrame and select specific columns directly
qa_data_df = qa_data.to_pandas()[['answer_en', 'question_en', 'context', 'context_data']]

# Drop rows with missing values in essential columns
qa_data_df = qa_data_df.dropna(subset=['answer_en', 'question_en', 'context', 'context_data'])

# Display the first few rows to confirm
print(qa_data_df.head())

# Save the preprocessed data if needed
qa_data_df.to_csv("processed_quran_qa_data.csv", index=False)