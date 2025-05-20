
import pandas as pd
import numpy as np
import os
import sys
import warnings
import contextlib
import cloudpickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / np.clip(denominator, 1e-8, None)) * 100


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

    text_features = ["title", "description_x"]
    categorical_features = ["formatted_experience_level", "state", "industry"]
    numeric_features = ["benefit_count", "skills_length", "year", "follower_count", "employee_count"]

    df[text_features] = df[text_features].fillna("")
    df[categorical_features] = df[categorical_features].fillna("missing")
    df[numeric_features] = df[numeric_features].fillna(0)

    X = df[text_features + categorical_features + numeric_features]
    y = np.log1p(df["normalized_salary"])

    preprocessor = ColumnTransformer(transformers=[
        ("tfidf_title", TfidfVectorizer(max_features=100, stop_words='english'), "title"),
        ("tfidf_desc", TfidfVectorizer(max_features=100, stop_words='english'), "description_x"),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ])

    base_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(random_state=42, verbose=-1))
    ])

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [4, 6],
        "model__subsample": [0.6, 0.8],
        "model__colsample_bytree": [0.6, 0.8]
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Running hyperparameter tuning...")
    with suppress_stderr():
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    model = best_pipeline.named_steps["model"]

    # Predict on test data
    y_pred = best_pipeline.predict(X_test)

    # Evaluate performance
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    percent_errors = np.abs((y_test - y_pred) / np.clip(y_test, 1e-8, None))
    median_pct_error = np.median(percent_errors) * 100

    print(f"Test MAE: {mae:.2f}")
    print(f"Median Absolute Percentage Error: {median_pct_error:.2f}%")


    # Extract feature names
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = []

    # Get feature names from each transformer
    for name, transformer, cols in preprocessor.transformers_:
        if name == "tfidf_title":
            feature_names += [f"tfidf_title__{w}" for w in transformer.get_feature_names_out()]
        elif name == "tfidf_desc":
            feature_names += [f"tfidf_desc__{w}" for w in transformer.get_feature_names_out()]
        elif name == "onehot":
            feature_names += transformer.get_feature_names_out(cols).tolist()
        elif name == "num":
            feature_names += cols

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nTop 20 Feature Importances:")
    print(feature_importance_df.head(20).to_string(index=False))

    # Save model
    os.makedirs("model", exist_ok=True)
    with open("model/salary_prediction_pipeline.pkl", "wb") as f:
        cloudpickle.dump(best_pipeline, f)

    print("Model pipeline saved using cloudpickle.")


if __name__ == "__main__":
    train_and_save_pipeline()
