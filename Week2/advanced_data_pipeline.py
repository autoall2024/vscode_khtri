#!/usr/bin/env python3
"""
Advanced Data Testing Pipeline:
1. Load or generate a small "real" dataset
2. Train a CTGAN model to generate synthetic data
3. Perform advanced anomaly detection with PyOD
4. Clean anomalies
5. Validate data with Great Expectations
6. Train & evaluate a RandomForest classifier
"""

import numpy as np
import pandas as pd

# CTGAN for synthetic data
from ctgan import CTGAN

# PyOD for anomaly detection
from pyod.models.iforest import IForest  # Alternatively, AutoEncoder, etc.

# scikit-learn for model training/evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# For data validation
import great_expectations as ge


# -----------------------------
# 1. Acquire or create a "real" dataset
# -----------------------------
def get_real_dataset():
    """
    In a real scenario, you'd load your production data.
    Here, we create a small synthetic dataset using scikit-learn
    just to have a 'real' dataset for demonstration.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=500, 
        n_features=8, 
        n_informative=5, 
        random_state=42
    )
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df_real = pd.DataFrame(X, columns=columns)
    df_real["target"] = y
    return df_real


# -----------------------------
# 2. Train CTGAN on the "real" dataset
# -----------------------------
def train_ctgan(df, target_col="target", epochs=5, batch_size=64):
    """
    Train a CTGAN model on the given tabular data (excluding the target).
    Return a trained CTGANSynthesizer.
    Note: In practice, you might want more epochs (>5) for better results.
    """
    # Separate features from target
    train_data = df.drop(columns=[target_col]).copy()
    discrete_cols = []  # List any categorical columns if you have them

    # Initialize CTGAN
    ctgan = CTGANSynthesizer(epochs=epochs, batch_size=batch_size)
    ctgan.fit(train_data, discrete_cols)

    return ctgan


# -----------------------------
# 3. Generate Synthetic Data with CTGAN
# -----------------------------
def generate_synthetic_data(ctgan_model, num_rows=1000):
    """
    Use the trained CTGAN model to sample new synthetic data.
    """
    synthetic_data = ctgan_model.sample(num_rows)
    return synthetic_data


# -----------------------------
# 4. Merge Synthetic Data with Real Target
#    (Or Synthesize Target Separately)
# -----------------------------
def combine_synthetic_features_with_target(synthetic_features, real_df, target_col="target"):
    """
    Since CTGAN was trained only on features (excluding the target),
    we can do one of the following:
      1) Synthesize the target as well (if we treat 'target' as categorical).
      2) Or keep the real dataset's target distribution.

    For simplicity, let's randomly sample from the real dataset's target
    distribution and attach to the synthetic features.
    """
    df_combined = synthetic_features.copy()
    # Randomly sample from real target distribution
    synthetic_targets = np.random.choice(real_df[target_col].unique(), size=len(df_combined))
    df_combined[target_col] = synthetic_targets
    return df_combined


# -----------------------------
# 5. Advanced Anomaly Detection (PyOD)
# -----------------------------
def detect_anomalies_with_iforest(df, target_col="target"):
    """
    Use IsolationForest from PyOD to detect anomalies in the feature columns.
    Return a boolean mask: True for "normal" rows, False for anomalies.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    
    # Initialize and fit
    clf = IForest(contamination=0.05, random_state=42)
    clf.fit(X)

    # Predict: 1 = normal, -1 = anomaly
    preds = clf.labels_
    mask_normal = (preds == 0)
    return mask_normal


def remove_anomalies(df, target_col="target"):
    """
    Detect anomalies and remove them from the dataset.
    """
    mask_normal = detect_anomalies_with_iforest(df, target_col=target_col)
    cleaned_df = df[mask_normal].copy()
    return cleaned_df


# -----------------------------
# 6. Data Validation (Great Expectations)
# -----------------------------
def validate_data(df, target_col="target"):
    """
    Run Great Expectations checks to ensure data meets basic criteria.
    For demonstration, we do:
      1) No nulls in feature columns
      2) target is in an expected set
      3) Numeric columns in a plausible range (just an example)
    """
    ge_df = ge.from_pandas(df)
    numeric_cols = [c for c in df.columns if c != target_col]

    # Expect no null values
    for col in numeric_cols:
        ge_df.expect_column_values_to_not_be_null(col)

    # Expect target to be within known range [0,1] for classification
    unique_targets = df[target_col].unique()
    ge_df.expect_column_values_to_be_in_set(target_col, set(unique_targets))

    # Expect numeric columns to be in a range (arbitrary for demonstration)
    # Could be refined if you know domain-specific constraints
    for col in numeric_cols:
        ge_df.expect_column_values_to_be_between(col, -100.0, 100.0)

    res = ge_df.validate()
    return res


# -----------------------------
# 7. Train & Evaluate (RandomForest)
# -----------------------------
def train_and_evaluate(df, target_col="target"):
    """
    Train a simple RandomForest to see how the data influences performance.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


# -----------------------------
# 8. End-to-End Pipeline
# -----------------------------
def end_to_end_workflow():
    print("=== Loading or Creating a Small 'Real' Dataset ===")
    df_real = get_real_dataset()
    print(f"Real dataset shape: {df_real.shape}")

    print("\n=== Training CTGAN on Real Dataset ===")
    ctgan_model = train_ctgan(df_real, target_col="target", epochs=5, batch_size=64)
    print("CTGAN training complete.")

    print("\n=== Generating Synthetic Data ===")
    df_synthetic_features = generate_synthetic_data(ctgan_model, num_rows=1000)
    print(f"Synthetic features shape: {df_synthetic_features.shape}")

    print("\n=== Combining Synthetic Features with Real Target Distribution ===")
    df_synthetic = combine_synthetic_features_with_target(df_synthetic_features, df_real)
    print(f"Synthetic dataset shape: {df_synthetic.shape}")

    print("\n=== Detecting and Removing Anomalies (IsolationForest) ===")
    df_synthetic_cleaned = remove_anomalies(df_synthetic, target_col="target")
    print(f"Shape after removing anomalies: {df_synthetic_cleaned.shape}")

    print("\n=== Validating Data with Great Expectations ===")
    validation_results = validate_data(df_synthetic_cleaned, target_col="target")
    print("Validation success:", validation_results.success)
    if not validation_results.success:
        print("Some expectations failed. See the details in validation_results for more info.")

    print("\n=== Training & Evaluating Model on Cleaned Synthetic Data ===")
    accuracy = train_and_evaluate(df_synthetic_cleaned, target_col="target")
    print(f"Accuracy on synthetic data (with anomalies removed): {accuracy:.4f}")


# -----------------------------
# Main Entry
# -----------------------------
if __name__ == "__main__":
    end_to_end_workflow()
