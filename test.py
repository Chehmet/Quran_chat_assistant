import joblib

def test_rag(question):
    # Load the saved pipeline
    pipeline = joblib.load("rag_pipeline.pkl")
    
    # Run the pipeline
    result = pipeline.run(
        query=question,
        params={
            "Retriever": {"top_k": 5},
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
for i in range(5):
    question = input()
    result = test_rag(question)
    print(result)