
import streamlit as st
import pandas as pd
import joblib

st.title('Stroke Prediction Demo')
model = joblib.load('best_stroke_model.joblib')

st.sidebar.header('Patient input')
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=45)
gender = st.sidebar.selectbox('Gender', ['Male','Female','Other'])
hypertension = st.sidebar.selectbox('Hypertension', [0,1])
heart_disease = st.sidebar.selectbox('Heart disease', [0,1])
ever_married = st.sidebar.selectbox('Ever married', ['Yes','No'])
work_type = st.sidebar.selectbox('Work type', ['children','Govt_job','Never_worked','Private','Self-employed'])
Residence_type = st.sidebar.selectbox('Residence type', ['Urban','Rural'])
avg_glucose_level = st.sidebar.number_input('Average glucose level', min_value=40.0, max_value=400.0, value=100.0)
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0)
smoking_status = st.sidebar.selectbox('Smoking status', ['never smoked','formerly smoked','smokes','Unknown'])

input_dict = {
    'gender':[gender], 'age':[age], 'hypertension':[hypertension], 'heart_disease':[heart_disease],
    'ever_married':[ever_married], 'work_type':[work_type], 'Residence_type':[Residence_type],
    'avg_glucose_level':[avg_glucose_level], 'bmi':[bmi], 'smoking_status':[smoking_status]
}
input_df = pd.DataFrame(input_dict)

if st.button('Predict'):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0,1] if hasattr(model, 'predict_proba') else None
    if pred==1:
        st.error(f'Prediction: Stroke likely (probability {proba:.2f})')
    else:
        st.success(f'Prediction: No stroke predicted (probability {proba:.2f})')
