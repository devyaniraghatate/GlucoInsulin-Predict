import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Biomarker Classification App", layout="centered")

st.title("🩺 Biomarker-Based Health Classification")
st.write('''
This app uses a K-Nearest Neighbors (KNN) model to predict clinical classification
(Control vs. Patient) based on physical and blood analysis biomarkers.
''')

st.sidebar.header("Input Biomarkers")

# Feature input fields
age = st.sidebar.slider("Age", min_value=10, max_value=100, value=50, step=1)
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=10.0, max_value=300.0, value=90.0)
insulin = st.sidebar.number_input("Insulin (µU/mL)", min_value=1.0, max_value=100.0, value=10.0)
homa = st.sidebar.number_input("HOMA", min_value=0.1, max_value=30.0, value=2.0)
leptin = st.sidebar.number_input("Leptin (ng/mL)", min_value=1.0, max_value=120.0, value=20.0)
adiponectin = st.sidebar.number_input("Adiponectin (µg/mL)", min_value=1.0, max_value=40.0, value=10.0)
resistin = st.sidebar.number_input("Resistin (ng/mL)", min_value=1.0, max_value=100.0, value=10.0)
mcp_1 = st.sidebar.number_input("MCP-1 (pg/mL)", min_value=10.0, max_value=2000.0, value=400.0)

# Create DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'bmi': bmi,
    'glucose': glucose,
    'insulin': insulin,
    'homa': homa,
    'leptin': leptin,
    'adiponectin': adiponectin,
    'resistin': resistin,
    'mcp_1': mcp_1
}])

st.subheader("User Input Features")
st.write(input_data)

if st.button("Predict"):
    # Scale the inputs using the loaded scaler
    scaled_input = scaler.transform(input_data)

    # Run prediction
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    st.subheader("Prediction Results")
    if prediction == 1:
        st.success("Prediction: Class 1 (Control Group)")
    else:
        st.error("Prediction: Class 2 (Patient Group)")

    st.write(f"**Confidence Level (Class 1 - Control):** {probabilities[0]:.2%}")
    st.write(f"**Confidence Level (Class 2 - Patient):** {probabilities[1]:.2%}")