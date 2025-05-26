import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('harddiseasemodel.pkl', 'rb'))

st.title("Heart Disease Prediction")

st.markdown("### Enter the patient's details below:")

# Collecting 13 input fields (excluding 'Target')
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FBS)", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (CA)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    # Collect the input data as a list
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Make prediction
    prediction = model.predict(input_data)

    # Show result
    if prediction[0] == 1:
        st.error("The prediction suggests that the person *has heart disease*. Please consult a doctor for a medical checkup.")
    else:
        st.success("Good news! The prediction suggests that the person *does NOT have heart disease*.")
