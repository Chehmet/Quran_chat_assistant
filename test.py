import joblib

# Function to test the RAG pipeline
def test_rag(question):
    # Load the saved pipeline
    pipeline = joblib.load("rag_pipeline2.pkl")
    
    # Run the pipeline
    result = pipeline.run(
        query=question,
        params={
            "BM25Retriever": {"top_k": 5},  # Adjust based on which retriever you want to test
            "Reader": {"top_k": 1}
        }
    )
    
    # Extract the answer and its metadata
    if result['answers']:
        answer = result['answers'][0].answer
        confidence = result['answers'][0].score
        context = result['answers'][0].context
        surah = result['answers'][0].meta.get('surah', 'Unknown')
        ayat = result['answers'][0].meta.get('ayat', 'Unknown')
        
        # Prepare the output
        output = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "context": context,
            "surah": surah,
            "ayat": ayat
        }
    else:
        output = {
            "question": question,
            "answer": "Sorry, I couldn't find an answer to your question.",
            "confidence": None,
            "context": None,
            "surah": None,
            "ayat": None
        }
    
    return output

# Example usage
for _ in range(5):
    question = input("Enter your question: ")
    result = test_rag(question)
    print(result)
