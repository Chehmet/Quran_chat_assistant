from rag_pipe_with_qa import * 


def test_rag(question):
    # Load the optimized pipeline
    pipeline = joblib.load("rag_pipeline_combined_optimized.pkl")
    
    # Check similarity
    most_similar_idx, similarity_score = get_most_similar_question(question)
    
    # Run query with tuned parameters regardless of similarity score
    result = pipeline.run(
        query=question,
        params={
            "Retriever": {"top_k": 15},  # Retrieve more passages for better coverage
            "Reader": {"top_k": 5}       # Increase reader top_k to find better answers
        }
    )
    
    # Extract the best answer with additional metadata
    if result['answers']:
        best_answer = max(result['answers'], key=lambda x: x.score)  # Pick the highest confidence answer
        
        answer = best_answer.answer.strip()  # Clean up whitespace from answer
        confidence = best_answer.score
        
        context = best_answer.context.strip()  # Clean up whitespace from context
        
        source_type = best_answer.meta.get('type', 'Unknown')
        surah = best_answer.meta.get('surah', 'Unknown') if source_type == 'tafseer' else None
        ayat = best_answer.meta.get('ayat', 'Unknown') if source_type == 'tafseer' else None
        
        # Prepare output with source-specific information and improved readability
        if source_type == 'tafseer':
            explanation = f"In Surah **{surah}**, Ayat **{ayat}**, it is mentioned: _{context}_."
        else:
            explanation = f"The QA dataset provided this answer based on context: _{context}_."
        
        output = {
            "question": question,
            "answer": answer,
            "confidence": round(confidence * 100, 2),  # Convert confidence to percentage for readability
            "explanation": explanation,
            "similarity_score": round(similarity_score * 100, 2)  # Show similarity score as a percentage too
        }
        
        return output
    
    else:
        return {
            "error": "No relevant answers found. Please try rephrasing your question."
        }

# Example usage:
if __name__ == "__main__":
    question_input = input("Type your question: ")
    response_output = test_rag(question_input)
    print(response_output)