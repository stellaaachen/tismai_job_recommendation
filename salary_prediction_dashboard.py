import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load trained pipeline ===
@st.cache_resource
def load_model():
    return joblib.load("model/salary_prediction_pipeline.pkl")

model = load_model()

# === UI Header ===
st.title("💼 Tech Job Salary Predictor")
st.markdown("📌 *Model trained on LinkedIn job postings (2023–2024)*")

# === U.S. States for Dropdown ===
us_states = [
    'missing', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL',
    'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM',
    'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN',
    'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

# === Input Form ===
with st.form("prediction_form"):
    title = st.text_input("Job Title", "Software Engineer")
    description = st.text_area("Job Description", "We are seeking a skilled software engineer with experience in Python, SQL, and cloud services.")

    experience_level = st.selectbox("Experience Level", ["missing", "Entry", "Mid", "Senior", "Executive"])
    state = st.selectbox("State", us_states, index=us_states.index("CA"))
    industry = st.selectbox("Industry", ["Tech", "Other"])

    benefit_count = st.number_input("Number of Benefits Mentioned", min_value=0, value=3)
    skills_length = st.number_input("Number of Skills Mentioned", min_value=0, value=10)
    year = st.selectbox("Year Listed", [2023, 2024], index=1)
    follower_count = st.number_input("Company Follower Count", min_value=0, value=10000)
    employee_count = st.number_input("Company Employee Count", min_value=0, value=500)

    submitted = st.form_submit_button("Predict Salary")

# === Prediction Logic ===
if submitted:
    input_df = pd.DataFrame([{
        "title": title,
        "description_x": description,
        "formatted_experience_level": experience_level,
        "state": state,
        "industry": industry,
        "benefit_count": benefit_count,
        "skills_length": skills_length,
        "year": year,
        "follower_count": follower_count,
        "employee_count": employee_count
    }])

    try:
        log_pred_salary = model.predict(input_df)[0]
        predicted_salary = np.expm1(log_pred_salary)

        st.success(f"💰 Predicted Normalized Salary: **${predicted_salary:,.2f}**")

    except Exception as e:
        st.error("⚠️ Something went wrong with the prediction.")
        st.exception(e)

