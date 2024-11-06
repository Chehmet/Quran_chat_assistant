import pandas as pd

# Load the dataset
file_path = 'main_df.csv'
df = pd.read_csv(file_path)

# Filter required columns
df = df[['Surah', 'Ayat', 'Tafaseer - Tafsir al-Jalalayn']]
df.columns = ['Surah', 'Ayat', 'Tafseer']

# Save the processed data for model input
df.to_csv('processed_quran_data.csv', index=False)
