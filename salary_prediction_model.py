import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
import joblib
import os
import warnings


warnings.filterwarnings("ignore")


# === Load and Merge Data ===
def load_and_prepare_data():
    jobs = pd.read_csv("data/IT_postings.csv")
    companies = pd.read_csv("data/companies.csv")
    employee_counts = pd.read_csv("data/employee_counts.csv")
    benefits = pd.read_csv("data/benefits.csv")

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

    df["state"] = df["location"].str.extract(r",\s*([A-Z]{2})")
    df["industry"] = df["description_y"].apply(lambda x: "Tech" if isinstance(x, str) and "software" in x.lower() else "Other")
    df["skills_length"] = df["skills_desc"].str.split().str.len()
    df["year"] = pd.to_datetime(df["listed_time"], errors='coerce').dt.year
    df["follower_count"].fillna(df["follower_count"].median(), inplace=True)
    df["employee_count"].fillna(df["employee_count"].median(), inplace=True)

    return df


def train_and_save_pipeline():
    df = load_and_prepare_data()
    df = df[df["normalized_salary"].notna()].copy()

    # === Define Features and Target ===
    text_features = ["title", "description_x"]
    categorical_features = ["formatted_experience_level", "state", "industry"]
    numeric_features = ["benefit_count", "skills_length", "year", "follower_count", "employee_count"]

    # Fill missing values
    df[text_features] = df[text_features].fillna("")
    df[categorical_features] = df[categorical_features].fillna("missing")
    df[numeric_features] = df[numeric_features].fillna(0)

    X = df[text_features + categorical_features + numeric_features]
    y = np.log1p(df["normalized_salary"])  # Log transform

    # === Define ColumnTransformer ===
    preprocessor = ColumnTransformer(transformers=[
        ("tfidf_title", TfidfVectorizer(max_features=100), "title"),
        ("tfidf_desc", TfidfVectorizer(max_features=100), "description_x"),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ])

    # === Define full pipeline (without model yet) ===
    base_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(random_state=42))
    ])

    # === Hyperparameter Grid ===
    param_grid = {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [4, 6],
        "model__subsample": [0.6, 0.8],
        "model__colsample_bytree": [0.6, 0.8]
    }

    # === Split and Search ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔍 Running hyperparameter tuning...")
    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    print("\n✅ Best Params:", search.best_params_)
    print(f"✅ Best CV MAE (log-space): {-search.best_score_:.4f}")

    # === Evaluate on test set ===
    best_pipeline = search.best_estimator_
    y_pred_log = best_pipeline.predict(X_test)
    y_pred_dollar = np.expm1(y_pred_log)
    y_true_dollar = np.expm1(y_test)

    mae = mean_absolute_error(y_true_dollar, y_pred_dollar)
    mape = mean_absolute_percentage_error(y_true_dollar, y_pred_dollar)

    print(f"\n💰 Test MAE: ${mae:,.2f}")
    print(f"📉 Test MAPE: {mape:.2f}%")

    # === Save model ===
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_pipeline, "model/salary_prediction_pipeline.pkl")
    print("📦 Model pipeline saved to model/salary_prediction_pipeline.pkl")


if __name__ == "__main__":
    train_and_save_pipeline()
