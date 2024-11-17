import streamlit as st
import requests

st.set_page_config(page_title="Quran QA App", page_icon="🌙", layout="centered")
st.title("🌙 Quran Answerer App")
st.markdown("Ask any question related to the Quran, and get precise answers using RAG-based retrieval.")
st.subheader("Ask a question and get insights from the Quran Tafseer.")

st.sidebar.title("Popular Questions")
popular_questions = ["Can I eat pork?", "Who refused Allah's command to prostrate to Adam (peace be upon him)?",
                      "Who raised the foundations of the Holy House (the Kaaba)?", "What is the significance of prayer in Islam?",
                      "Who are the people not allowed to fast during the month of Ramadan?", 
                      "What is the evidence for Allah (SWT) multiplying the reward for charity in His path to seven hundred times?",
                      "What is the true religion in the sight of God?"]

selected_question = st.sidebar.radio("Choose a question:", popular_questions, index=0)
custom_question = st.text_input("Or ask your own question here:")

question = custom_question if custom_question else selected_question

st.write("### Your Question:")
st.write(question)

if st.button("Get answer"):
    if question.strip() == "":
        st.warning("Please enter a valid question.")
    with st.spinner("Fetching answer..."):
        response = requests.post("http://127.0.0.1:8000/get_advice", json={"question": question})
        if response.status_code == 200:
            data = response.json()
            if data['advice'] != "Your question is not similar enough to the available dataset.":
                st.write("### Short answer:")
                st.write(data['advice'])
                st.write("### Context:")
                st.write(data['context'])
            else:
                st.error("No answer found. Question is not similar enough to the dataset.")
        else:
            st.error("Error fetching advice.")