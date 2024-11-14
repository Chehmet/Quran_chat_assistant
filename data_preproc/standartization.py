import pandas as pd
import json
import ast

# Load the datasets
dataset1 = pd.read_csv('datasets/processed_quran_data.csv')
dataset2 = pd.read_csv('datasets/processed_quran_qa_data.csv')

# Parse `context_data` in dataset2, assuming it contains JSON-like structures
dataset2['context_data'] = dataset2['context_data'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Extract relevant columns from `context_data` if needed
def extract_surah_ayah(context_data):
    if isinstance(context_data, list):
        return [(item.get('surah'), item.get('ayah')) for item in context_data]
    return None

dataset2['surah_ayah'] = dataset2['context_data'].apply(extract_surah_ayah)

# Example to merge or concatenate datasets based on Surah and Ayat if relevant
# dataset_combined = pd.merge(dataset1, dataset2, left_on=['Surah', 'Ayat'], right_on=[...], how='outer')

print(dataset1.head())
print(dataset2.head())
