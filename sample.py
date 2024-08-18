import streamlit as st

st.title("🌟 Surya's Portfolio ChatBot 🤖")
st.header("Welcome to Surya's Portfolio!")
st.write("Hi! 👋 My name is Sia. I'm here to help answer your questions about Surya's portfolio.")
st.write("Try asking me questions like:")
st.write("* 💻 How is this application built? What technologies were used in this project?")
st.write("* 📚 What projects has Surya worked on?")
st.write("* 💡 What are Surya's skills and expertise?")
st.write("* 📊 Can you summarize Surya's experience in a specific industry?")
st.write("or anything else you're curious about Surya's portfolio!")

# User input field with a placeholder
user_query = st.text_input("Ask me a question...", placeholder="Type your question here...")

# Button to trigger the chatbot response with a custom label
if st.button("Get Answer 🚀"):
    response = handle_query(user_query)
    st.write(response)