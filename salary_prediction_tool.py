

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
import joblib

# === Load Data ===
jobs = pd.read_csv("data/job_postings.csv")
companies = pd.read_csv("data/companies.csv")
employee_counts = pd.read_csv("data/employee_counts.csv")
benefits = pd.read_csv("data/benefits.csv")

# === Merge Datasets ===
df = jobs.merge(companies, on="company_id", how="left")
df = df.merge(
    employee_counts.groupby("company_id").agg({
        "employee_count": "mean",
        "follower_count": "mean"
    }), on="company_id", how="left"
)
benefit_counts = benefits.groupby("job_id").size().reset_index(name="benefit_count")
df = df.merge(benefit_counts, on="job_id", how="left")
df["benefit_count"] = df["benefit_count"].fillna(0)

# === Drop NAs and Prepare Features ===
df = df.dropna(subset=["med_salary", "skills_desc", "formatted_experience_level", "location"])
df = df[df["med_salary"] < 1000000]  # remove outliers

# Extract state from location
df["state"] = df["location"].str.extract(r",\s*([A-Z]{2})")

# Simplified industry column
df["industry"] = df["description_y"].apply(lambda x: "Tech" if isinstance(x, str) and "software" in x.lower() else "Other")

# === Features and Labels ===
X = df[["skills_desc", "formatted_experience_level", "state", "industry", "benefit_count"]]
y = df["med_salary"]

# === Preprocessing Pipeline ===
skills_vectorizer = TfidfVectorizer(max_features=100)
categorical_features = ["formatted_experience_level", "state", "industry"]
numerical_features = ["benefit_count"]

preprocessor = ColumnTransformer(
    transformers=[
        ("skills", skills_vectorizer, "skills_desc"),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# === Final Pipeline ===
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# === Evaluate ===
y_pred = model_pipeline.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")

# === Save Artifacts ===
joblib.dump(model_pipeline, "salary_model.pkl")
