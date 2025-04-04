import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
    # Combine text sequence and categorical feature
    features_combined = np.concatenate([text_input, type_input], axis=1)
    prediction = model.predict(features_combined)
    
    # priority prediction (0 = High, 1 = Low, 2 = Medium)
    predicted_priority = np.argmax(prediction, axis=1)[0]  # Get the predicted priority label (0, 1, 2)
    
    # Map the predicted priority to human-readable label
    return priority_mapping[predicted_priority]

# Simple UI 
st.title("Resolve AI")
st.write("Enter your request and select a type to generate a prediction.")

# User input fields (the two features the model was trained on)
user_input = st.text_area("Enter your text:", "")
type_selection = st.selectbox("Select type:", type_options)

# Prediction UI functionality
if st.button("Generate Prediction"):
    if user_input:
        # Preprocess inputs
        text_input = preprocess_text(user_input)
        type_input = preprocess_type(type_selection)

        # Get prediction
        predicted_priority = generate_prediction(text_input, type_input)

        # Display result
        st.write(f"Predicted priority: {predicted_priority}")
    else:
        st.error("Please enter some text!")