import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model (replace with your actual .h5 file path)
model = tf.keras.models.load_model("../modeling/artifacts/best_model.h5")

# Function to process the input text
def preprocess_text(text, type_value):
    # Example: You can modify this to match your model's input preprocessing
    # For example, converting text to lowercase, tokenizing, etc.
    text_input = np.array([text.lower()])  # Example preprocessing (change as needed)
    
    # You can include the 'type' in preprocessing if necessary
    type_input = np.array([type_value])  # Assuming type_value is numerical or needs to be encoded
    
    return text_input, type_input

# Function to generate predictions
def generate_prediction(text_input, type_input):
    # Make the prediction using the loaded model
    prediction = model.predict([text_input, type_input])  # Adjust this line according to your model's input
    return prediction

# Streamlit UI elements
st.title("Resolve AI")
st.write("Enter your request and select a type to get a priority prediction.")

# Free form text input
user_input = st.text_area("Enter your text:", "")

# Dropdown for selecting "type"
type_options =  ['Change' 'Incident' 'Problem' 'Request'] # Modify as per your options
type_selection = st.selectbox("Select type:", type_options)

# Convert type selection to a numerical value (or however you want to encode it)
type_mapping = {type_options[i]: i for i in range(len(type_options))}
type_value = type_mapping[type_selection]

# Prediction button
if st.button("Generate Prediction"):
    if user_input:
        # Preprocess input and type
        text_input, type_input = preprocess_text(user_input, type_value)
        
        # Get prediction
        prediction = generate_prediction(text_input, type_input)
        
        # Display result (you can modify how the prediction is displayed)
        st.write("Prediction result: ", prediction)
    else:
        st.error("Please enter some text!")