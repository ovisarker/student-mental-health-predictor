import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="wide")

# --- Load Models and Encoders ---
@st.cache_resource
def load_resources():
    try:
        encoders = joblib.load('models/encoders.joblib')
        stress_model = joblib.load('models/stress_model.joblib')
        anxiety_model = joblib.load('models/anxiety_model.joblib')
        depression_model = joblib.load('models/depression_model.joblib')
        return encoders, stress_model, anxiety_model, depression_model
    except FileNotFoundError:
        return None, None, None, None

encoders, stress_model, anxiety_model, depression_model = load_resources()

if not all([encoders, stress_model, anxiety_model, depression_model]):
    st.error("Model or encoder files not found! Please run the 'train_models.py' script first.")
    st.stop()

# --- App Title and Description ---
st.title("🧠 Student Mental Health Prediction System")
st.markdown("This application predicts a student's risk level for Stress, Anxiety, and Depression using Machine Learning.")
st.markdown("**Disclaimer:** This is not a medical diagnosis. It's a tool to help identify potential risks early.")
st.markdown("---")

# --- Input Form in Columns ---
st.header("Please Fill Out Your Information and Survey")
col1, col2 = st.columns(2)

# Column 1: Demographics and PSS
with col1:
    st.subheader("Your Information")
    age = st.slider("Age", 18, 40, 21)
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    university = st.selectbox("University", encoders['University'].classes_)
    department = st.selectbox("Department", encoders['Department'].classes_)
    academic_year = st.selectbox("Academic Year", encoders['Academic Year'].classes_)
    cgpa = st.selectbox("Current CGPA", encoders['Current CGPA'].classes_)
    scholarship = st.selectbox("Waiver or Scholarship", encoders['waiver_or_scholarship'].classes_)

    st.subheader("Stress Survey (PSS-10)")
    st.write("*In the last month, how often have you...*")
    pss_scores = [st.slider(f"PSS Question {i+1}", 0, 4, 2) for i in range(10)]

# Column 2: GAD and PHQ
with col2:
    st.subheader("Anxiety Survey (GAD-7)")
    st.write("*Over the last 2 weeks, how often have you been bothered by...*")
    gad_scores = [st.slider(f"GAD Question {i+1}", 0, 3, 1) for i in range(7)]

    st.subheader("Depression Survey (PHQ-9)")
    st.write("*Over the last 2 weeks, how often have you been bothered by...*")
    phq_scores = [st.slider(f"PHQ Question {i+1}", 0, 3, 1) for i in range(9)]

# --- Prediction Logic ---
if st.button("Analyze and Predict Status", type="primary"):
    # 1. Create a dictionary with all user inputs
    user_data = {
        'Age': age, 'Gender': gender, 'University': university, 'Department': department,
        'Academic Year': academic_year, 'Current CGPA': cgpa, 'waiver_or_scholarship': scholarship
    }
    
    # Add survey scores
    for i, score in enumerate(pss_scores): user_data[f'PSS{i+1}'] = score
    for i, score in enumerate(gad_scores): user_data[f'GAD{i+1}'] = score
    for i, score in enumerate(phq_scores): user_data[f'PHQ{i+1}'] = score

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # 3. Preprocess the data using loaded encoders
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # 4. Make Predictions
    stress_pred = stress_model.predict(input_df[stress_model.feature_names_in_])[0]
    anxiety_pred = anxiety_model.predict(input_df[anxiety_model.feature_names_in_])[0]
    depression_pred = depression_model.predict(input_df[depression_model.feature_names_in_])[0]

    # --- Display Results ---
    st.markdown("---")
    st.header("📈 Prediction Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Stress Level", stress_pred)
    res_col2.metric("Anxiety Level", anxiety_pred)
    res_col3.metric("Depression Level", depression_pred)

    # --- Display Tiered Support ---
    st.markdown("---")
    st.subheader("Recommended Support")
    if "High" in stress_pred or "Severe" in anxiety_pred or "Severe" in depression_pred:
        st.error("The results indicate a significant level of distress. It is strongly recommended to seek professional help from your university's counseling service or a trusted professional.")
    elif "Moderate" in stress_pred or "Moderate" in anxiety_pred or "Moderate" in depression_pred:
        st.warning("The results suggest a moderate level of distress. Proactively managing your mental health can be very beneficial. Consider reaching out to university support services.")
    else:
        st.success("The results indicate a low level of distress. Continue to focus on wellness and building healthy coping strategies.")