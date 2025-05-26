# heart_disease_backend.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib  # or use pickle

# Load the trained model
model = joblib.load("heart_disease_model.pkl")  # Change path if needed

# Page title
st.set_page_config(page_title="Heart Disease Prediction")

st.title("Heart Disease Prediction System")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (CP)", options=[0, 1, 2, 3])
bps = st.number_input("Rest Blood Pressure (BPS)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fps = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FPS)", options=[0, 1])
rest = st.selectbox("Resting ECG", options=[0, 1, 2])
cg = st.number_input("Cardiac Gradient (CG)", min_value=60, max_value=250, value=150)
talash = st.number_input("TALASH", min_value=0.0, max_value=10.0, value=1.0)
exchange = st.number_input("Exchange", min_value=0.0, max_value=10.0, value=1.0)
old_peaks = st.number_input("Old Peaks", min_value=0.0, max_value=10.0, value=0.5)
slope = st.selectbox("SLOPE", options=[0, 1, 2])
ca = st.selectbox("CA (Number of Major Vessels)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("THAL", options=[0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, bps, chol, fps, rest, cg, talash,
                            exchange, old_peaks, slope, ca, thal]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("Prediction: High likelihood of heart disease.")
        st.warning("Based on the input parameters, there is a significant risk of heart disease. It is strongly recommended to consult a healthcare professional for further evaluation and diagnosis.")
    else:
        st.success("Prediction: Low likelihood of heart disease.")
        st.info("No major risk detected. Nonetheless, regular medical checkups and a healthy lifestyle are advised.")
