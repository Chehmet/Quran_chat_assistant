from datasets import load_dataset
import pandas as pd

# Load the dataset and specify the split
ds = load_dataset("M-AI-C/quran_tafseer", split="train")

# Convert to a DataFrame and select specific columns
df = ds.to_pandas()[['sorah', 'ayah', 'en-sarwar']]
df.columns = ['Surah', 'Ayat', 'Tafseer']

# Save the processed data for model input
df.to_csv('processed_quran_data.csv', index=False)
