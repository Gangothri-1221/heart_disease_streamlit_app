import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('heart_model.pkl', 'rb'))

st.title("Heart Disease Prediction App")

# User input fields
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', value=120)
chol = st.number_input('Serum Cholestoral in mg/dl (chol)', value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting ECG results (restecg)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST depression (oldpeak)', value=1.0)
slope = st.selectbox('Slope of the peak exercise ST segment (slope)', [0, 1, 2])
ca = st.selectbox('Number of major vessels (ca)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

# Prediction button
if st.button('Predict'):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal)
    input_data_np = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_np)

    if prediction[0] == 0:
        st.success("The Person does NOT have Heart Disease.")
    else:
        st.error("The Person HAS Heart Disease.")
