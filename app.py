import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import uuid

# Load the trained SVM model, scaler, and label encoder
best_svm = joblib.load('best_svm_model.pkl')  # Ensure you save the model from the notebook
scaler = joblib.load('scaler.pkl')  # Ensure you save the scaler from the notebook
le = joblib.load('label_encoder.pkl')  # Ensure you save the label encoder from the notebook

# Function to process input features
def process_features(age, systolic_bp, diastolic_bp, cholesterol):
    # Create a DataFrame with the input features
    data = {
        'age': [age],
        'systolic_bp': [systolic_bp],
        'diastolic_bp': [diastolic_bp],
        'cholesterol': [cholesterol]
    }
    X = pd.DataFrame(data)

    # Add derived features as done in the notebook
    X['bp_ratio'] = X['systolic_bp'] / X['diastolic_bp'].replace(0, np.nan)
    X['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
    X['age_cholesterol'] = X['age'] * X['cholesterol']
    X['bp_product'] = X['systolic_bp'] * X['diastolic_bp']
    X['age_squared'] = X['age'] ** 2
    X['high_cholesterol'] = (X['cholesterol'] > 100).astype(int)

    # Handle outliers using IQR as done in the notebook
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # Impute NaNs with median
    X = X.fillna(X.median())

    # Standardize features
    X_scaled = scaler.transform(X)

    return X_scaled

# Streamlit UI
st.set_page_config(page_title="Diabetic Retinopathy Prediction", page_icon="🩺", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #1e3a8a;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #374151;
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 30px;
    }
    .input-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .prediction-text {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .probability-text {
        font-size: 1.2em;
        color: #4b5563;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1em;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
    }
    .stNumberInput label {
        color: #374151;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title">Diabetic Retinopathy Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your health metrics to predict the likelihood of diabetic retinopathy.</div>', unsafe_allow_html=True)

# Input form
with st.form(key='retinopathy_form'):
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=50.0, step=0.1)
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, max_value=300.0, value=120.0, step=0.1)
    
    with col2:
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0, step=0.1)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=500.0, value=200.0, step=0.1)
    
    submit_button = st.form_submit_button(label="Predict")
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if submit_button:
    # Process the input features
    X_scaled = process_features(age, systolic_bp, diastolic_bp, cholesterol)
    
    # Make prediction
    prediction = best_svm.predict(X_scaled)[0]
    prediction_proba = best_svm.predict_proba(X_scaled)[0]
    
    # Decode the prediction
    prediction_label = le.inverse_transform([prediction])[0]
    probability = max(prediction_proba) * 100
    
    # Display results
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    if prediction_label.lower() == 'retinopathy':
        st.markdown('<div class="prediction-text" style="color: #dc2626;">Positive for Diabetic Retinopathy (Retinopathy)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-text" style="color: #16a34a;">Negative for Diabetic Retinopathy (No Retinopathy)</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="probability-text">Confidence: {probability:.2f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #6b7280;">
        Powered by Streamlit | Model trained using SVM for diabetic retinopathy prediction
    </div>
""", unsafe_allow_html=True)