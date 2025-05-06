# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("salary_model.pkl")

# === Streamlit UI ===
st.title("💼 LinkedIn Salary Estimator")
st.markdown("Estimate your salary based on skills, experience, and location.")

# --- User Input ---
skills_input = st.text_area("Enter your skills (comma-separated):", "Python, SQL, Machine Learning")
experience_level = st.selectbox("Experience Level:", [
    "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"
])
state_input = st.selectbox("State (2-letter code):", [
    "CA", "NY", "TX", "IL", "WA", "MA", "FL", "Other"
])
industry_input = st.selectbox("Industry:", ["Tech", "Other"])
benefit_count = st.slider("How many benefits does the job include?", 0, 10, 3)

# --- Submit ---
if st.button("Estimate Salary"):
    user_input = pd.DataFrame({
        "skills_desc": [skills_input],
        "formatted_experience_level": [experience_level],
        "state": [state_input],
        "industry": [industry_input],
        "benefit_count": [benefit_count]
    })

    try:
        prediction = model.predict(user_input)[0]
        st.success(f"💰 Estimated Salary: ${int(prediction):,} per year")
    except Exception as e:
        st.error("Something went wrong with the prediction. Please check your input.")
        st.exception(e)

st.caption("Built with LinkedIn job data from 2023–2024")
