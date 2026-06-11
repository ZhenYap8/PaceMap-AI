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

# Columns screened for outliers using the Three Sigma Rule (~0.3% per variable)
OUTLIER_COLUMNS = [
    "distance_km",
    "elevation_gain_m",
    "avg_pace_s_per_km",
    "finish_time_s",
]

# Pace & finish-time focused cleaning for Learn My Runs
PACE_TIME_OUTLIER_COLUMNS = [
    "avg_pace_s_per_km",
    "finish_time_s",
    "implied_pace_s_per_km",
]

DEFAULT_SIGMA = 3.0
MIN_SAMPLES_FOR_OUTLIER_REMOVAL = 8
MIN_SAMPLES_PACE_TIME_CLEANING = 5

# Physical bounds — reject before statistical cleaning
MIN_PACE_S_PER_KM = 120.0    # 2:00 /km
MAX_PACE_S_PER_KM = 1200.0   # 20:00 /km
MIN_DISTANCE_KM = 0.1
MIN_FINISH_TIME_S = 60.0
MAX_PACE_TIME_MISMATCH_RATIO = 0.65  # implied vs recorded avg pace


def remove_outliers_three_sigma(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    sigma: float = DEFAULT_SIGMA,
    id_column: Optional[str] = "run_id",
    min_rows: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove statistical outliers using the Three Sigma Rule.

    A row is flagged if any checked column lies outside mean ± sigma·std.
    Under a normal distribution ~0.3% of values fall beyond 3σ per variable.

    Skipped when the dataset is too small for reliable estimates.
    Never removes so many rows that fewer than ``min_rows`` remain.

    Args:
        df: DataFrame containing run features.
        columns: Numeric columns to check (defaults to OUTLIER_COLUMNS).
        sigma: Number of standard deviations (default 3.0).
        id_column: Optional column used to identify removed runs in the report.
        min_rows: Minimum rows to retain after cleaning.

    Returns:
        Tuple of (cleaned DataFrame, outlier report dict).
    """
    columns = columns or OUTLIER_COLUMNS
    columns = [c for c in columns if c in df.columns]
    n_before = len(df)

    report: dict = {
        "method": "three_sigma",
        "sigma": sigma,
        "columns_checked": columns,
        "total_before": n_before,
        "total_after": n_before,
        "removed_count": 0,
        "removed_pct": 0.0,
        "skipped": False,
        "outliers": [],
    }

    if n_before < MIN_SAMPLES_FOR_OUTLIER_REMOVAL or not columns:
        report["skipped"] = True
        report["skip_reason"] = (
            f"Need at least {MIN_SAMPLES_FOR_OUTLIER_REMOVAL} runs for outlier detection"
            if n_before < MIN_SAMPLES_FOR_OUTLIER_REMOVAL
            else "No valid columns to check"
        )
        return df.copy(), report

    outlier_mask = pd.Series(False, index=df.index)
    z_scores_by_col: dict[str, pd.Series] = {}

    for col in columns:
        series = df[col].astype(float)
        mean = series.mean()
        std = series.std()
        if std < 1e-9:
            continue
        z = ((series - mean) / std).abs()
        z_scores_by_col[col] = z
        outlier_mask |= z > sigma

    if not outlier_mask.any():
        return df.copy(), report

    # Remove outliers but keep at least min_rows (drop most extreme first)
    outlier_indices = df.index[outlier_mask].tolist()
    if n_before - len(outlier_indices) < min_rows:
        # Rank by max z-score across columns, remove only the worst offenders
        max_z = pd.Series(0.0, index=df.index)
        for z in z_scores_by_col.values():
            max_z = pd.concat([max_z, z], axis=1).max(axis=1)
        n_to_remove = max(0, n_before - min_rows)
        if n_to_remove == 0:
            return df.copy(), report
        worst = max_z.nlargest(n_to_remove).index.tolist()
        outlier_mask = df.index.isin(worst)
        outlier_indices = worst

    cleaned = df.loc[~outlier_mask].copy()

    for idx in outlier_indices:
        triggered = []
        for col, z in z_scores_by_col.items():
            if z.loc[idx] > sigma:
                label = _COLUMN_LABELS.get(col, col)
                value = _format_metric(col, float(df.loc[idx, col]))
                triggered.append(f"{label} {value} (z={z.loc[idx]:.2f})")
        entry = {"reasons": triggered}
        if id_column and id_column in df.columns:
            entry["run_id"] = df.loc[idx, id_column]
        report["outliers"].append(entry)

    n_after = len(cleaned)
    report["total_after"] = n_after
    report["removed_count"] = n_before - n_after
    report["removed_pct"] = round((n_before - n_after) / n_before * 100, 2)

    logger.info(
        f"Three-sigma outlier removal: {report['removed_count']} of {n_before} "
        f"runs removed ({report['removed_pct']}%)."
    )
    return cleaned, report


def _format_metric(col: str, value: float) -> str:
    """Format a run metric for human-readable outlier messages."""
    from pacemap.utils import format_duration, format_pace

    if col in ("avg_pace_s_per_km", "implied_pace_s_per_km"):
        return format_pace(value)
    if col == "finish_time_s":
        return format_duration(value)
    if col == "distance_km":
        return f"{value:.2f} km"
    if col == "elevation_gain_m":
        return f"{value:.0f} m"
    return f"{value:.2g}"


_COLUMN_LABELS = {
    "avg_pace_s_per_km": "Avg pace",
    "finish_time_s": "Finish time",
    "implied_pace_s_per_km": "Implied pace",
    "distance_km": "Distance",
    "elevation_gain_m": "Elevation",
}


def _add_implied_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Derive pace from finish time ÷ distance for consistency checks."""
    df = df.copy()
    df["implied_pace_s_per_km"] = np.where(
        df["distance_km"] > 0,
        df["finish_time_s"] / df["distance_km"],
        0.0,
    )
    return df


def _outlier_entry(
    df: pd.DataFrame,
    idx: int,
    reasons: list[str],
    id_column: Optional[str],
) -> dict:
    """Build a human-readable outlier report entry."""
    from pacemap.utils import format_duration, format_pace

    entry: dict = {"reasons": reasons}
    if id_column and id_column in df.columns:
        entry["run_id"] = df.loc[idx, id_column]
    if "avg_pace_s_per_km" in df.columns:
        entry["avg_pace"] = format_pace(float(df.loc[idx, "avg_pace_s_per_km"]))
    if "finish_time_s" in df.columns:
        entry["finish_time"] = format_duration(float(df.loc[idx, "finish_time_s"]))
    if "distance_km" in df.columns:
        entry["distance_km"] = round(float(df.loc[idx, "distance_km"]), 2)
    return entry


def filter_invalid_pace_time_runs(
    df: pd.DataFrame,
    id_column: Optional[str] = "run_id",
) -> tuple[pd.DataFrame, dict]:
    """
    Remove runs with impossible or inconsistent pace/time values.

    Catches GPS glitches, stopped watches, and corrupt GPX timestamps.
    """
    df = _add_implied_pace(df)
    n_before = len(df)
    invalid_indices: list[int] = []
    invalid_entries: list[dict] = []

    from pacemap.utils import format_duration, format_pace

    for idx, row in df.iterrows():
        reasons: list[str] = []
        dist = float(row.get("distance_km", 0))
        pace = float(row.get("avg_pace_s_per_km", 0))
        finish = float(row.get("finish_time_s", 0))
        implied = float(row.get("implied_pace_s_per_km", 0))

        if dist < MIN_DISTANCE_KM:
            reasons.append(f"Distance too short ({dist:.2f} km)")
        if finish < MIN_FINISH_TIME_S:
            reasons.append(f"Finish time too short ({format_duration(finish)})")
        if pace <= 0 or pace < MIN_PACE_S_PER_KM:
            reasons.append(f"Avg pace unrealistically fast ({format_pace(pace)})")
        if pace > MAX_PACE_S_PER_KM:
            reasons.append(f"Avg pace unrealistically slow ({format_pace(pace)})")
        if dist >= 0.5 and pace > 0:
            mismatch = abs(implied - pace) / pace
            if mismatch > MAX_PACE_TIME_MISMATCH_RATIO:
                reasons.append(
                    f"Pace/time mismatch — recorded {format_pace(pace)} vs "
                    f"implied {format_pace(implied)} ({mismatch * 100:.0f}% off; "
                    f"likely bad GPS or timestamps)"
                )

        if reasons:
            invalid_indices.append(idx)
            invalid_entries.append(_outlier_entry(df, idx, reasons, id_column))

    cleaned = df.drop(index=invalid_indices).copy() if invalid_indices else df.copy()

    report = {
        "stage": "invalid_pace_time",
        "description": "Ignored runs with impossible or inconsistent pace/time",
        "removed_count": len(invalid_indices),
        "ignored_runs": invalid_entries,
    }
    logger.info(
        f"Invalid pace/time filter: removed {report['removed_count']} of {n_before} runs."
    )
    return cleaned, report


def clean_runs_for_learning(
    df: pd.DataFrame,
    id_column: Optional[str] = "run_id",
    sigma: float = DEFAULT_SIGMA,
    min_rows: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline for Learn My Runs.

    1. Drop invalid pace/time (physical bounds + consistency)
    2. Drop statistical outliers on pace, finish time, and implied pace (3σ rule)
    """
    n_uploaded = len(df)
    stages: list[dict] = []

    # Stage 1 — hard invalid pace/time filter (always runs)
    df, invalid_report = filter_invalid_pace_time_runs(df, id_column=id_column)
    if invalid_report["removed_count"]:
        stages.append(invalid_report)

    if len(df) < min_rows:
        return df, {
            "method": "pace_time_cleaning",
            "sigma": sigma,
            "total_before": n_uploaded,
            "total_after": len(df),
            "removed_count": n_uploaded - len(df),
            "removed_pct": round((n_uploaded - len(df)) / n_uploaded * 100, 2) if n_uploaded else 0,
            "skipped": True,
            "skip_reason": "Too few valid runs remaining after invalid filter",
            "stages": stages,
            "outliers": [r for s in stages for r in s.get("ignored_runs", [])],
        }

    # Stage 2 — Three Sigma on pace & time columns
    pace_time_report: dict = {
        "stage": "pace_time_three_sigma",
        "description": "Ignored runs with pace or finish time beyond 3σ",
        "removed_count": 0,
        "ignored_runs": [],
    }

    if len(df) >= MIN_SAMPLES_PACE_TIME_CLEANING:
        pre_sigma = df.copy()
        df, sigma_report = remove_outliers_three_sigma(
            df,
            columns=PACE_TIME_OUTLIER_COLUMNS,
            sigma=sigma,
            id_column=id_column,
            min_rows=min_rows,
        )
        if sigma_report["removed_count"]:
            pace_time_report["removed_count"] = sigma_report["removed_count"]
            pace_time_report["sigma"] = sigma_report["sigma"]
            for o in sigma_report["outliers"]:
                entry: dict = {"run_id": o.get("run_id"), "reasons": o["reasons"]}
                if o.get("run_id") and id_column and id_column in pre_sigma.columns:
                    match = pre_sigma[pre_sigma[id_column] == o["run_id"]]
                    if not match.empty:
                        entry = _outlier_entry(
                            pre_sigma, match.index[0], readable, id_column
                        )
                pace_time_report["ignored_runs"].append(entry)
            stages.append(pace_time_report)
    else:
        stages.append({
            "stage": "pace_time_three_sigma",
            "skipped": True,
            "skip_reason": (
                f"Need at least {MIN_SAMPLES_PACE_TIME_CLEANING} runs for 3σ pace/time cleaning"
            ),
        })

    all_ignored = []
    for stage in stages:
        all_ignored.extend(stage.get("ignored_runs", []))

    full_report = {
        "method": "pace_time_cleaning",
        "sigma": sigma,
        "total_before": n_uploaded,
        "total_after": len(df),
        "removed_count": n_uploaded - len(df),
        "removed_pct": round((n_uploaded - len(df)) / n_uploaded * 100, 2) if n_uploaded else 0,
        "skipped": False,
        "stages": stages,
        "outliers": all_ignored,
    }
    logger.info(
        f"Pace/time cleaning complete: {full_report['removed_count']} of {n_uploaded} "
        f"runs ignored ({full_report['removed_pct']}%)."
    )
    return df, full_report


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
            df[col].fillna(median_val, inplace=True)
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
    id_col = "run_id" if "run_id" in df.columns else None
    df, _ = clean_runs_for_learning(df, id_column=id_col)
    df = preprocess_features(df.drop(columns=["run_id", "implied_pace_s_per_km"], errors="ignore"))
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
