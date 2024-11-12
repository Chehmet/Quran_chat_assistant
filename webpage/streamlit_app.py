import streamlit as st
import requests

st.title("Quran Advice App")
st.write("Ask a question and get advice from the Quran Tafseer.")

user_input = st.text_input("Ask for advice:")

if st.button("Get Advice"):
    response = requests.post("http://127.0.0.1:8000/get_advice", json={"question": user_input})
    if response.status_code == 200:
        data = response.json()
        st.write("### Advice:")
        st.write(data['advice'])
        st.write("### Context:")
        st.write(data['context'])
    else:
        st.error("Error fetching advice.")
