import joblib

# Function to test the RAG pipeline
def test_rag(question, use_dpr=False):
    # Load the correct pipeline based on `use_dpr`
    pipeline_name = "rag_pipeline_dpr.pkl" if use_dpr else "rag_pipeline_bm25.pkl"
    pipeline = joblib.load(pipeline_name)
    
    # Run the pipeline
    result = pipeline.run(
        query=question,
        params={
            "Retriever": {"top_k": 5},  # Adjust based on the retriever you want to test
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
    use_dpr = input("Use DensePassageRetriever? (y/n): ").strip().lower() == "y"
    result = test_rag(question, use_dpr)
    print(result)
