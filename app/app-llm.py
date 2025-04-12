import os
import streamlit as st
import random
import time

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import pandas as pd 

from sklearn.model_selection import train_test_split

# Download dataset
file_path = "dataset-tickets-multi-lang-4-20k.csv"
df = pd.read_csv(file_path)

# Pre-processing of the dataset to prepare for VectorDB creation
df = df[df['language'] == 'en']
non_string_body = df[~df['body'].apply(lambda x: isinstance(x, str))].index
non_string_answers = df[~df['answer'].apply(lambda x: isinstance(x, str))].index
non_string_ids = non_string_body.union(non_string_answers)
df = df.drop(index=non_string_ids)
df['q_and_a'] = 'Question: ' + df['body'] + ' Answer: ' + df['answer']
df_train, df_holdout = train_test_split(df, test_size=0.2, random_state=42)

# Setup of chromadb database
persist_directory = './chroma_db'
loader = DataFrameLoader(
    df_train,
    page_content_column="q_and_a")
documents = loader.load()

# Get OpenAI setup
openai_api_key = os.getenv("openai_token")

# Cache the creation of chroma_db so it only runs at app startup
@st.cache_resource
def get_vectordb():
    embedding = OpenAIEmbeddings(openai_api_key=os.getenv("openai_token"))
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory)

vectordb = get_vectordb()


llm_name = "gpt-3.5-turbo"

llm = ChatOpenAI(model_name=llm_name, temperature=0.7,
                 openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5})
)

# Emulate a streamed response
def response_generator(prompt):
    response = qa_chain({"query": prompt})['result']   
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Technical Support Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your question here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
