import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import time
from langchain.vectorstores import Chroma
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the trained model
model = tf.keras.models.load_model("../modeling/artifacts/best_model.h5")

# Load the tokenizer
with open("../modeling/artifacts/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define fixed categories for 'type'
type_options = ["Change", "Incident", "Problem", "Request"]

# Define hardcoded label mapping for encoded results
priority_mapping = {0: "High", 1: "Low", 2: "Medium"}

# Constants
MAX_LENGTH = 512  

# Function to preprocess text input
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    return padded_sequence

# Function to preprocess categorical input (type)
def preprocess_type(selected_type):
    mapping = {val: idx for idx, val in enumerate(type_options)}
    return np.array([[mapping[selected_type]]])

# Function to make predictions
def generate_prediction(text_input, type_input):
    features_combined = np.concatenate([text_input, type_input], axis=1)
    prediction = model.predict(features_combined)
    predicted_priority = np.argmax(prediction, axis=1)[0]  # Get the predicted priority label (0, 1, 2)
    return priority_mapping[predicted_priority]

# OpenAI setup
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
persist_directory = './chroma_db'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

def response_generator(prompt):
    response = qa_chain({"query": prompt})['result']   
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Streamlit UI
st.title("Resolve AI")
st.write("Enter your request and select a type to generate a prediction.")

user_input = st.text_area("Enter your text:", "")
type_selection = st.selectbox("Select type:", type_options)

if st.button("Generate Prediction"):
    if user_input:
        text_input = preprocess_text(user_input)
        type_input = preprocess_type(type_selection)
        predicted_priority = generate_prediction(text_input, type_input)
        
        st.write(f"Predicted priority: {predicted_priority}")
        
        if predicted_priority in ["Medium", "High"]:
            st.warning("This issue may require human intervention. Please contact support.")
        else:
            st.subheader("Chat Assistance")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("How can I assist you further?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(prompt))
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please enter some text!")
