Portfolio ChatBot is a cutting-edge conversational AI project that leverages vector search 
and natural language processing to provide users with accurate and informative responses. 
The project consists of six key components:
1. Database Creation: Astra DB is used to store and manage the vector database.
2. PDF Processing: Text is extracted from a collection of PDF files using pdfminer.six.
3. Vector Database: The extracted text is embedded and loaded into Astra DB using the 
Python client.
4. Chatbot Development: A chatbot is built using LLaMA 3.1 and LangChain, featuring a 
prompt template that matches user queries to the vector database.
5. Web Development: A website is created using Streamlit to provide a user friendly interface for interacting with the chatbot.
6. Deployment: The website is deployed using Hugging Face Spaces. 
7. HuggingFace Spaces Link : [https://huggingface.co/spaces/myaijournal/portfolio_chatbot]
