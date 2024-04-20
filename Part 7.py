import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from lime.lime_text import LimeTextExplainer

filepath = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)

# Define function to load data with caching
@st.cache_data
def load_data(fpath):
    df = pd.read_csv(fpath)
    return df

# Define function to load ML model with caching
@st.cache_resource
def load_model_ml(fpath):
    return joblib.load(fpath)

# App Title
st.title('Review Rating Prediction')


df = load_data(fpath=FPATHS['data']['raw']['full'])

lstm_model = load_model_ml(fpath=FPATHS['models']['lstm_model'])

# Review
user_review = st.text_area('Enter your review:', '')

# To get a model prediction
def get_prediction(text):
    text= Lemmas-joined
    prediction = lstm_model.predict(np.array([text]))
    return prediction[0]  

# Function to explain prediction using Lime Text Explainer
def explain_prediction(text):
    explainer = LimeTextExplainer()
    explanation = explainer.explain_instance(text, lstm_model.predict)
    return explanation

# Prediction button
if st.button('Get Prediction'):
    prediction = get_prediction(user_review)
    st.write('Predicted Rating:', prediction)

# Checkbox for explanation
explain = st.checkbox('Include Explanation')

# If explanation checkbox is checked, show explanation
if explain:
    explanation = explain_prediction(user_review)
    st.write('Explanation:')
    # Display explanation as text or in some formatted way

# Option to load training and test data
if st.button('Load Data'):
    # Assuming 'ds' is your dataset containing text and ratings
    split_train = 0.7
    n_train_samples = int(len(ds) * split_train)
    # Calculate the number of samples for validation
    split_val = 0.2
    n_val_samples = int(len(ds) * split_val)  # Validation size
    # Test size is remainder
    split_test = 1 - (split_train + split_val)
    # Use .take to slice out the number of samples for training
    train_ds = ds.take(n_train_samples)
    # Skip over the training batches
    val_ds = ds.skip(n_train_samples)
    # Take .take to slice out the correct number of samples for validation
    val_ds = val_ds.take(n_val_samples)
    # Skip over all of the training + validation samples, the rest remain as samples for testing
    test_ds = ds.skip(n_train_samples + n_val_samples)

    # Assuming X_train, X_test, y_train, y_test are needed for evaluation
    # You may need to convert your dataset to X and y format depending on your implementation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test, random_state=42)

    st.write('Data Loaded Successfully!')

