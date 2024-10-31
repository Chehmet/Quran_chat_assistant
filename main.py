import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from chromadb import Client as ChromaClient
from chromadb.config import Settings

# Initialize ChromaDB
client = ChromaClient()
collection = client.create_collection("faq_support")

# Sample FAQ data for customer support
faq_data = [
    {"question": "How do I reset my password?", "answer": "To reset your password, go to the settings page and select 'Reset Password'."},
    {"question": "What is the refund policy?", "answer": "Our refund policy allows returns within 30 days of purchase with proof of receipt."},
    {"question": "How do I contact support?", "answer": "You can contact support via email at support@example.com or by calling our hotline."}
]

# Insert FAQ data into ChromaDB
for i, faq in enumerate(faq_data):
    collection.add(ids=[str(i)], documents=[faq['question']], metadatas=[faq])

# Load the language model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def retrieve_faq(question):
    # Retrieve the most relevant FAQ from ChromaDB based on similarity
    results = collection.query(query_texts=[question], n_results=1)
    
    # Check if results is non-empty and contains expected data
    if results and isinstance(results, list) and len(results) > 0:
        # Access the first result's metadata
        metadatas = results[0].get("metadatas", [])
        if metadatas:
            return metadatas[0]["answer"]
    return None



def customer_support_chatbot(user_query):
    retrieved_answer = retrieve_faq(user_query)

    # If a relevant FAQ answer exists, augment it with LLM for customization
    if retrieved_answer:
        input_text = f"Customer asked: '{user_query}'\nSupport answer based on FAQ: '{retrieved_answer}'\nComplete response:"
    else:
        input_text = f"Customer asked: '{user_query}'\nResponse:"

    # Generate response using the language model
    response = qa_pipeline(input_text, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]

# Example usage
user_question = "Can I get a refund if I don't like the product?"
response = customer_support_chatbot(user_question)
print("Bot:", response)
