"""
ml_model.py

Responsible for:
- Preprocessing
- Feature engineering
- Model training
- Prediction
- Evaluation
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Supported model types
MODEL_LINEAR = "linear"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_GRADIENT_BOOSTING = "gradient_boosting"

SUPPORTED_MODELS = [MODEL_LINEAR, MODEL_RANDOM_FOREST, MODEL_GRADIENT_BOOSTING]

# Expected feature columns
FEATURE_COLUMNS = [
    "distance_km",
    "elevation_gain_m",
    "avg_pace_s_per_km",
    "avg_heart_rate_bpm",
    "temperature_c",
    "time_of_day_hour",
    "fatigue_score",
]

TARGET_COLUMN = "finish_time_s"


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate a DataFrame of activity features.

    - Drops rows where the target column is missing.
    - Fills missing numeric features with column medians.
    - Validates expected feature columns are present.

    Args:
        df: Raw DataFrame containing activity data.

    Returns:
        Cleaned DataFrame ready for feature engineering.

    Raises:
        ValueError: If required columns are entirely absent.
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    df = df.copy()

    # Drop rows where target is unknown
    before = len(df)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing target values.")

    # Impute missing features with median to avoid data leakage across splits
    for col in FEATURE_COLUMNS:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"Imputed '{col}' with median={median_val:.2f}.")

    logger.info(f"Preprocessed dataset: {len(df)} rows retained.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the DataFrame.

    New features:
        - pace_per_elevation: avg_pace_s_per_km / (elevation_gain_m + 1)
        - distance_x_fatigue: distance_km * fatigue_score

    Args:
        df: Preprocessed DataFrame.

    Returns:
        DataFrame with additional engineered columns.
    """
    df = df.copy()
    df["pace_per_elevation"] = df["avg_pace_s_per_km"] / (
        df["elevation_gain_m"] + 1
    )
    df["distance_x_fatigue"] = df["distance_km"] * df["fatigue_score"]
    logger.info("Feature engineering complete.")
    return df


def build_model(model_type: str = MODEL_LINEAR) -> Pipeline:
    """
    Build a scikit-learn Pipeline for the chosen model type.

    Preferred order (README):
        1. Linear Regression
        2. Random Forest
        3. Gradient Boosting

    Args:
        model_type: One of 'linear', 'random_forest', 'gradient_boosting'.

    Returns:
        A scikit-learn Pipeline (scaler + estimator).

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == MODEL_LINEAR:
        estimator = LinearRegression()
    elif model_type == MODEL_RANDOM_FOREST:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == MODEL_GRADIENT_BOOSTING:
        estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {SUPPORTED_MODELS}"
        )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    logger.info(f"Built pipeline with model: {model_type}")
    return pipeline


def train_model(
    df: pd.DataFrame,
    model_type: str = MODEL_LINEAR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train a model on the provided DataFrame.

    Performs preprocessing, feature engineering, train/test split,
    and model fitting.

    Args:
        df: Raw activity DataFrame.
        model_type: Model to use (default: linear regression).
        test_size: Fraction of data to reserve for testing.
        random_state: Reproducibility seed.

    Returns:
        Tuple of (fitted_pipeline, X_train, y_train, X_test, y_test).
    """
    df = preprocess_features(df)
    df = engineer_features(df)

    feature_cols = FEATURE_COLUMNS + ["pace_per_elevation", "distance_x_fatigue"]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_model(model_type)
    pipeline.fit(X_train, y_train)

    logger.info(
        f"Model trained on {len(X_train)} samples | "
        f"Test set: {len(X_test)} samples."
    )
    return pipeline, X_train, y_train, X_test, y_test


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a trained model pipeline on the test set.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        X_test: Test feature DataFrame.
        y_test: True target values.

    Returns:
        Dictionary with MAE, RMSE, and R² metrics.
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }

    logger.info(
        f"Evaluation → MAE: {metrics['mae']:.2f}s | "
        f"RMSE: {metrics['rmse']:.2f}s | R²: {metrics['r2']:.4f}"
    )
    return metrics


def predict(
    pipeline: Pipeline,
    features: dict,
) -> float:
    """
    Make a single prediction using a trained pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        features: Dictionary of feature name → value. Must include all
                  FEATURE_COLUMNS plus engineered features.

    Returns:
        Predicted finish time in seconds.
    """
    input_df = pd.DataFrame([features])

    # Apply same feature engineering
    input_df["pace_per_elevation"] = input_df["avg_pace_s_per_km"] / (
        input_df["elevation_gain_m"] + 1
    )
    input_df["distance_x_fatigue"] = (
        input_df["distance_km"] * input_df["fatigue_score"]
    )

    prediction = pipeline.predict(input_df)[0]
    logger.info(f"Predicted finish time: {prediction:.1f}s")
    return float(prediction)


# --- Example usage ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200

    sample_data = pd.DataFrame(
        {
            "distance_km": rng.uniform(5, 42, n),
            "elevation_gain_m": rng.uniform(0, 500, n),
            "avg_pace_s_per_km": rng.uniform(240, 420, n),
            "avg_heart_rate_bpm": rng.uniform(130, 180, n),
            "temperature_c": rng.uniform(5, 30, n),
            "time_of_day_hour": rng.uniform(5, 20, n),
            "fatigue_score": rng.uniform(0, 10, n),
        }
    )

    # Synthetic target: distance * avg_pace + noise
    sample_data[TARGET_COLUMN] = (
        sample_data["distance_km"] * sample_data["avg_pace_s_per_km"]
        + rng.normal(0, 60, n)
    )

    for model_type in SUPPORTED_MODELS:
        print(f"\n=== {model_type.upper()} ===")
        pipeline, X_train, y_train, X_test, y_test = train_model(
            sample_data, model_type=model_type
        )
        metrics = evaluate_model(pipeline, X_test, y_test)
        print(
            f"MAE: {metrics['mae']:.1f}s | "
            f"RMSE: {metrics['rmse']:.1f}s | "
            f"R²: {metrics['r2']:.4f}"
        )
