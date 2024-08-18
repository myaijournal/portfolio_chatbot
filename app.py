import cassio
from langchain_openai import ChatOpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_community.chat_models import ChatOpenAI

import streamlit as st

from dotenv import load_dotenv
import os

from PyPDF2 import PdfReader

def extract_text_from_pdfs(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(directory, filename)
            with open(pdf_file_path, 'rb') as pdf_file_obj:
                pdf_reader = PdfReader(pdf_file_obj)
                num_pages = len(pdf_reader.pages)
                text = ''
                for page in range(num_pages):
                    page_obj = pdf_reader.pages[page]
                    text += page_obj.extract_text()
            yield filename, text

# Specify the directory containing the PDFs
directory = 'portfolio_pdfs'

# Initialize an empty string to accumulate all text
all_text = ''

# Use the generator to accumulate text from all PDFs
for filename, text in extract_text_from_pdfs(directory):
    all_text += text

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.5)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID) # Initialize CassandraDB

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

astra_vectorstore = Cassandra(
    embedding=embedding, table_name="portfolio_vectordb", session=None, keyspace=None)

astra_vectorstore.add_texts([all_text[:2000]])

class CustomVectorStoreIndexWrapper(VectorStoreIndexWrapper):
    def __init__(self, vectorstore):
        super().__init__(vectorstore)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vectorstore)

# # Define the prompt template
# prompt_template = """
# Given the context: {query}, generate a concise response. You are an AI assistant who answers user queries based on the provided search context. If you don't find a perfect match, politely punt.
# """
# user_template = "{query}"

# chatprompt = ChatPromptTemplate.from_messages([
#     ("system", prompt_template),
#     ("human", user_template)
# ])

# # Define the output parser
# class CleanOutput(BaseOutputParser):
#     def parse(self, output):
#         return output.strip()

# # Initialize the chain
# chain = chatprompt|llm|CleanOutput()

# def handle_query(query):
#     # Search the Vector DB for the most similar embeddings
#     answer = astra_vector_index.query(query, llm=llm).strip()

#     # Run the sequential chain
#     response = chain.invoke(relevant_text)

#     return response


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
    answer = astra_vector_index.query(user_query, llm=llm).strip()
    st.write(answer)
