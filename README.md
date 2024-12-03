# Quran-Based Chat/Advisor

This project is a conversational assistant that uses the Quran for answers, built using the Haystack library, FAISS document storage, and Uvicorn for FastAPI service. The assistant leverages embeddings to provide relevant answers from indexed content in response to user queries.
## Response from db
![](example/example_with_front.png)

## Response from rag
![](example/image_from_rag.png)

### Features

- **Document Retrieval**: Efficient retrieval of Quranic [tafseer](https://huggingface.co/datasets/M-AI-C/quran_tafseer) (I use Sarwar, as it's the most accurate according to my studies) through the FAISS document store.
- **Questions and answers dataset**: [This dataset](https://huggingface.co/datasets/nazimali/quran-question-answer-context) contains 1224 questions with answers from Quran, so If your question is in this dataset, answer will be more accurate.
- **Embedding-Based Search**: Uses embeddings to ensure contextually accurate responses to queries.
- **Multiprocessing and FastAPI**: Utilizes Uvicorn’s multiprocessing and asynchronous capabilities to handle concurrent requests.

### Installation

To set up the project, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/ATLLM_chat_assistant.git
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Make sure you have TensorFlow installed for embedding processing. In the future, I plan to upload a `.pkl` file to the cloud, but this is still under consideration:
   ```bash
   pip install tensorflow
   ```

### Project Structure

- `api.py` - Main FastAPI application script.
- `requirements.txt` - Contains the list of dependencies.
- `data/` - Folder to store documents and FAISS indices.
- `config/` - Configuration files for database and FAISS indexing.
- `test.py` - For console-based testing.

---

### Setup and Usage
 
1. **Run the Application:**
   Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn api:app --reload
   ```
2. **Run Streamlit:**
   ```bash
   streamlit run streamlit_app.py
   ```
3. **Access the API:**
   The server should be running at `http://127.0.0.1:8000`.

### Future Improvements

- Add Hadiths, not just Quranic content.
- Improve accuracy by refining responses, possibly through fine-tuning.
- Add visualizations or an interactive UI to enhance usability for users.

### Conract information:
If you have any questions you can contact me via [telegram](https://t.me/Chehmet)
