import cassio
from langchain_openai import ChatOpenAI
#from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatOpenAI

import streamlit as st

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough


######

#Function to load pdfs
def extract_text_from_pdfs(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(directory, filename)
            pdf_reader = PyPDFLoader(pdf_file_path)
            docs = pdf_reader.load()
            docs.extend(pdf_reader.load())
######

# Specify the directory containing the PDFs
directory = 'portfolio_pdfs'
docs = []

extract_text_from_pdfs(directory)

######

load_dotenv() #load env variables

ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

######

#define llm and embeddings
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=400)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

######

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID) # Initialize CassandraDB

######

#Splitting docs into chucks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

######

#initialising vector store object
astra_vectorstore = Cassandra.from_documents(
    documents=documents,
    embedding=embedding, 
    table_name="portfolio_vectordb")


######

# Index wrapping for faster query
#class CustomVectorStoreIndexWrapper(VectorStoreIndexWrapper):
    # def __init__(self, vectorstore):
    #     super().__init__(vectorstore)

#astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vectorstore)

######

# Define the prompt template

prompt = ChatPromptTemplate.from_template("""
    
    You are an AI assistant who answers user queries about Surya's Portfolio. 
                                          
    Generate a concise response only using the following pieces of retrieved text to answer the question. 
                                          
    If you don't find a perfect match, politely punt.
                                                                   
    Question: {query}
                                          
    Context: {context}                                  
                                          
    Answer:
                                          
                                          """)

######

# Retrieve and generate

retriever = astra_vectorstore.as_retriever(search_kwargs={"k": 2})

######

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create document chain
chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

######

# Streamlit interface

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
    answer = chain.invoke(user_query)
    st.write(answer)
