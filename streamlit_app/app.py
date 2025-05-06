# streamlit_app/app.py
import streamlit as st
import pandas as pd
import pickle
import json

# Load model pipeline
with open('../models/Heart_Disease.pkl', 'rb') as f:
    model = pickle.load(f)

# Load features
with open('../models/features.json', 'r') as f:
    features = json.load(f)

# UI title
st.title("Heart Disease Prediction")

# Input form
user_input = {}
st.header("Enter Patient Details")

# Define options for categorical fields
categorical_options = {
    "Sex": ["0", "1"],
    "ChestPainType": ["1", "2", "3", "4"],
    "FastingBS": ["0", "1"],
    "RestingECG": ["0", "1", "2"],
    "ExerciseAngina": ["0", "1"],
    "ST_Slope": ["1", "2", "3"],
    "Thal": ["3", "6", "7"]
}

for feature in features:
    if feature in categorical_options:
        user_input[feature] = st.selectbox(feature, categorical_options[feature])
    else:
        user_input[feature] = st.number_input(feature)

# Create input DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Presence of Heart Disease" if prediction == 1 else "No Heart Disease"
    st.success(f"Prediction: {result}")
