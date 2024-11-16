import streamlit as st
import requests

# Set page title and layout
st.set_page_config(page_title="Quran Answerer", layout="centered")

st.title("🌙 Quran Answerer App")
st.subheader("Ask a question and get insights from the Quran Tafseer.")

# Sidebar for popular questions
st.sidebar.title("Popular Questions")
popular_questions = ["Can I eat pork?", "Who refused Allah's command to prostrate to Adam (peace be upon him)?",
                      "Who raised the foundations of the Holy House (the Kaaba)?", "What is the significance of prayer in Islam?",
                      "Who are the people not allowed to fast during the month of Ramadan?", 
                      "What is the evidence for Allah (SWT) multiplying the reward for charity in His path to seven hundred times?",
                      "What is the true religion in the sight of God?"]

# User can select a popular question or enter their own
selected_question = st.sidebar.radio("Choose a question:", popular_questions, index=0)
custom_question = st.text_input("Or ask your own question here:")

# Use the custom question if entered, otherwise use the selected popular question
question = custom_question if custom_question else selected_question

# Display the current question
st.write("### Your Question:")
st.write(question)

if st.button("Get answer"):
    response = requests.post("https://quran-chat-assistant.streamlit.app/get_advice", json={"question": question})
    print(response.status_code)
    print(response.text)  
    if response.status_code == 200:
        data = response.json()
        if data['advice'] != "Your question is not similar enough to the available dataset.":
            st.write("### Short answer:")
            st.write(data['advice'])
            st.write("### Context:")
            st.write(data['context'])
        else:
            st.error("Question not similar enough to the dataset.")
    else:
        st.error("Error fetching advice.")
