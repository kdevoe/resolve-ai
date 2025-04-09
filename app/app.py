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
model = tf.keras.models.load_model("../modeling/artifacts/best_binary_model_after_tuning.h5")

# Load the tokenizer
with open("../modeling/artifacts/binary_tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define fixed categories for 'type'
type_options = ["Change", "Incident", "Problem", "Request"]

# Define hardcoded label mapping for encoded results
priority_mapping = {0: "Low", 1: "Med/High"}

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
    prediction = model.predict(features_combined)[0][0]  # Get the probability
    predicted_label = int(prediction > 0.5)  # Convert to 0 or 1
    return priority_mapping[predicted_label]

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
        
        if predicted_priority == "Med/High":
            st.warning("This issue may require human intervention. Please contact support.")
        else:
           chatbot_link = 'https://huggingface.co/spaces/kdevoe/TechSupportChatbot'
           st.write('Please chat with our [assistant](%s) for further resolution'% chatbot_link)
