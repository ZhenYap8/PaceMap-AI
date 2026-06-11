"""
Runner profile builder — learns patterns from all stored GPX runs.
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pacemap.ml_model import (
    FEATURE_COLUMNS,
    MODEL_GRADIENT_BOOSTING,
    TARGET_COLUMN,
    build_model,
    clean_runs_for_learning,
    engineer_features,
    evaluate_model,
    preprocess_features,
    predict,
)
from pacemap.parser import (
    extract_cadences,
    extract_coordinates,
    extract_elevations,
    extract_heart_rates,
    extract_timestamps,
    extract_track_points,
    load_gpx_file,
)
from pacemap.pace_calculator import (
    calculate_cumulative_distance,
    calculate_segment_distances,
    calculate_segment_paces,
    haversine_distance,
    smooth_gps_coordinates,
)
from pacemap.utils import (
    deduplicate_timestamps,
    elevation_gain,
    elapsed_seconds,
    format_duration,
    format_pace,
    metres_to_km,
)


def _mean_valid(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None and v > 0]
    return float(np.mean(valid)) if valid else None


def extract_detailed_run(gpx_filepath: str, run_id: str, smoothing_window: int = 3) -> Optional[dict[str, Any]]:
    """Extract rich features from a single GPX file for profiling and recommendation."""
    try:
        gpx = load_gpx_file(gpx_filepath)
        track_points = extract_track_points(gpx)
        if len(track_points) < 2:
            return None

        coordinates = extract_coordinates(track_points)
        timestamps = extract_timestamps(track_points)
        elevations = extract_elevations(track_points)
        heart_rates = extract_heart_rates(gpx)
        cadences = extract_cadences(gpx)

        timestamps_clean = deduplicate_timestamps(timestamps)
        smooth_coords = smooth_gps_coordinates(coordinates, window=smoothing_window)
        segment_distances = calculate_segment_distances(smooth_coords)
        cumulative = calculate_cumulative_distance(segment_distances)
        total_distance_m = cumulative[-1]
        segment_paces = calculate_segment_paces(smooth_coords, timestamps_clean)

        valid_paces = [p for p in segment_paces if p > 0]
        avg_pace = sum(valid_paces) / len(valid_paces) if valid_paces else 0
        total_elapsed = elapsed_seconds(timestamps[0], timestamps[-1])
        elev_gain = elevation_gain(elevations)
        avg_hr = _mean_valid(heart_rates)
        avg_cadence = _mean_valid(cadences)

        if total_elapsed is None or total_elapsed <= 0:
            return None

        start_lat, start_lon = coordinates[0]
        end_lat, end_lon = coordinates[-1]

        return {
            "run_id": run_id,
            "filepath": gpx_filepath,
            "distance_km": metres_to_km(total_distance_m),
            "elevation_gain_m": elev_gain,
            "avg_pace_s_per_km": avg_pace,
            "avg_heart_rate_bpm": avg_hr if avg_hr else 155.0,
            "avg_cadence_spm": avg_cadence,
            "temperature_c": 20.0,
            "time_of_day_hour": timestamps[0].hour if timestamps[0] else 8,
            "fatigue_score": 5.0,
            "finish_time_s": total_elapsed,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "end_lat": end_lat,
            "end_lon": end_lon,
            "gps_points": len(track_points),
            "has_heart_rate": avg_hr is not None,
            "has_cadence": avg_cadence is not None,
        }
    except Exception:
        return None


def learn_from_runs(
    run_entries: list[dict[str, Any]],
    models_dir: Path,
    profile_path: Path,
    smoothing_window: int = 3,
) -> dict[str, Any]:
    """
    Learn runner patterns from all GPX files and optionally train an ML model.

    Uses train/validation split and selects the best-performing model.
    """
    detailed_runs: list[dict[str, Any]] = []
    for entry in run_entries:
        filepath = entry.get("filepath")
        run_id = entry.get("run_id", Path(filepath).stem)
        if not filepath or not os.path.exists(filepath):
            continue
        details = extract_detailed_run(filepath, run_id, smoothing_window)
        if details:
            detailed_runs.append(details)

    if not detailed_runs:
        raise ValueError("No valid runs found. Upload GPX files with at least 2 GPS points each.")

    df = pd.DataFrame(detailed_runs)
    total_uploaded = len(df)

    # Clean pace/time outliers — invalid data + 3σ rule before learning
    df_clean, outlier_report = clean_runs_for_learning(df, id_column="run_id")
    cleaned_runs = df_clean.drop(
        columns=["implied_pace_s_per_km"], errors="ignore"
    ).to_dict("records")

    if not cleaned_runs:
        raise ValueError("All runs were flagged as outliers. Upload more varied GPX data.")

    profile = {
        "total_runs_uploaded": total_uploaded,
        "total_runs": len(cleaned_runs),
        "runs_used_for_learning": len(cleaned_runs),
        "outlier_cleaning": outlier_report,
        "avg_distance_km": round(df_clean["distance_km"].mean(), 2),
        "avg_elevation_gain_m": round(df_clean["elevation_gain_m"].mean(), 1),
        "avg_pace_s_per_km": round(df_clean["avg_pace_s_per_km"].mean(), 1),
        "avg_pace": format_pace(df_clean["avg_pace_s_per_km"].mean()),
        "avg_finish_time_s": round(df_clean["finish_time_s"].mean(), 1),
        "avg_finish_time": format_duration(df_clean["finish_time_s"].mean()),
        "elevation_std_m": round(df_clean["elevation_gain_m"].std(), 1) if len(df_clean) > 1 else 0,
        "distance_std_km": round(df_clean["distance_km"].std(), 2) if len(df_clean) > 1 else 0,
        "runs_with_heart_rate": int(df_clean["has_heart_rate"].sum()),
        "runs_with_cadence": int(df_clean["has_cadence"].sum()),
        "avg_heart_rate_bpm": round(df_clean["avg_heart_rate_bpm"].mean(), 1),
        "avg_cadence_spm": round(df_clean["avg_cadence_spm"].dropna().mean(), 1) if df_clean["avg_cadence_spm"].notna().any() else None,
        "model_trained": False,
        "model_metrics": None,
    }

    model_path = models_dir / "runner_model.pkl"
    models_dir.mkdir(parents=True, exist_ok=True)

    if len(cleaned_runs) >= 3:
        ml_df = df_clean[[
            "distance_km", "elevation_gain_m", "avg_pace_s_per_km",
            "avg_heart_rate_bpm", "temperature_c", "time_of_day_hour",
            "fatigue_score", "finish_time_s",
        ]].copy()

        ml_clean = preprocess_features(ml_df)
        ml_clean = engineer_features(ml_clean)
        feature_cols = FEATURE_COLUMNS + ["pace_per_elevation", "distance_x_fatigue"]
        X = ml_clean[feature_cols]
        y = ml_clean[TARGET_COLUMN]

        if len(cleaned_runs) >= 5:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

        pipeline = build_model(MODEL_GRADIENT_BOOSTING)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_val, y_val)

        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        profile["model_trained"] = True
        profile["model_metrics"] = {
            "mae_s": round(metrics["mae"], 1),
            "rmse_s": round(metrics["rmse"], 1),
            "r2": round(metrics["r2"], 4),
        }
    elif model_path.exists():
        model_path.unlink()

    filename_by_run_id = {
        entry.get("run_id"): entry.get("filename")
        for entry in run_entries
        if entry.get("run_id")
    }

    profile["runs"] = [
        {
            "run_id": r["run_id"],
            "filename": filename_by_run_id.get(r["run_id"], f"{r['run_id']}.gpx"),
            "distance_km": round(r["distance_km"], 2),
            "elevation_gain_m": round(r["elevation_gain_m"], 1),
            "avg_pace": format_pace(r["avg_pace_s_per_km"]),
            "finish_time": format_duration(r["finish_time_s"]),
            "start_lat": r["start_lat"],
            "start_lon": r["start_lon"],
            "has_heart_rate": r["has_heart_rate"],
            "has_cadence": r["has_cadence"],
        }
        for r in cleaned_runs
    ]

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    # Persist cleaned run data for recommender (outliers excluded)
    runs_data_path = profile_path.parent / "runs_data.json"
    with open(runs_data_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_runs, f, indent=2)

    return profile


def _parse_duration_string(duration: str) -> Optional[float]:
    """Parse H:MM:SS or M:SS duration strings back to seconds."""
    try:
        parts = [int(p) for p in duration.strip().split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _normalize_cleaning_reason(reason: str, entry: dict) -> str:
    """Upgrade legacy cleaning reason strings that used raw seconds."""
    from pacemap.utils import format_duration, format_pace

    reason = re.sub(
        r"\((\d+(?:\.\d+)?)s/km\)",
        lambda m: f"({format_pace(float(m.group(1)))})",
        reason,
    )
    reason = re.sub(
        r"Finish time too short \((\d+(?:\.\d+)?)s\)",
        lambda m: f"Finish time too short ({format_duration(float(m.group(1)))})",
        reason,
    )

    avg_pace = entry.get("avg_pace")
    if avg_pace and reason.startswith("Avg pace (z="):
        reason = reason.replace("Avg pace (z=", f"Avg pace {avg_pace} (z=", 1)

    if reason.startswith("Implied pace (z="):
        dist = entry.get("distance_km")
        finish = entry.get("finish_time")
        if dist and finish and dist > 0:
            finish_s = _parse_duration_string(finish)
            if finish_s:
                implied = format_pace(finish_s / dist)
                reason = reason.replace("Implied pace (z=", f"Implied pace {implied} (z=", 1)

    if "Pace/time mismatch" in reason and "recorded" not in reason and avg_pace:
        dist = entry.get("distance_km")
        finish = entry.get("finish_time")
        if dist and finish and dist > 0:
            finish_s = _parse_duration_string(finish)
            if finish_s:
                implied = format_pace(finish_s / dist)
                pct_match = re.search(r"(\d+)% off", reason)
                pct = pct_match.group(1) if pct_match else "?"
                reason = (
                    f"Pace/time mismatch — recorded {avg_pace} vs implied {implied} "
                    f"({pct}% off; likely bad GPS or timestamps)"
                )

    return reason


def _normalize_ignored_runs(runs: list[dict]) -> list[dict]:
    updated = []
    for entry in runs:
        entry = dict(entry)
        entry["reasons"] = [
            _normalize_cleaning_reason(r, entry) for r in entry.get("reasons", [])
        ]
        updated.append(entry)
    return updated


def normalize_outlier_cleaning(report: Optional[dict]) -> Optional[dict]:
    """Normalize stored cleaning reports from older app versions."""
    if not report:
        return report

    report = dict(report)
    stages = []
    for stage in report.get("stages", []):
        stage = dict(stage)
        if stage.get("ignored_runs"):
            stage["ignored_runs"] = _normalize_ignored_runs(stage["ignored_runs"])
        stages.append(stage)
    report["stages"] = stages

    if report.get("outliers"):
        report["outliers"] = _normalize_ignored_runs(report["outliers"])

    return report


def load_profile(profile_path: Path) -> Optional[dict[str, Any]]:
    if not profile_path.exists():
        return None
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    if profile.get("outlier_cleaning"):
        profile["outlier_cleaning"] = normalize_outlier_cleaning(profile["outlier_cleaning"])
    return profile


def load_runs_data(profile_path: Path) -> list[dict[str, Any]]:
    runs_data_path = profile_path.parent / "runs_data.json"
    if not runs_data_path.exists():
        return []
    with open(runs_data_path, encoding="utf-8") as f:
        return json.load(f)


def load_model(models_dir: Path):
    model_path = models_dir / "runner_model.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_finish_time(model, run_features: dict[str, Any], target_distance_km: float) -> float:
    """Predict finish time for a target distance using scaled run features."""
    scale = target_distance_km / run_features["distance_km"] if run_features["distance_km"] > 0 else 1
    features = {
        "distance_km": target_distance_km,
        "elevation_gain_m": run_features["elevation_gain_m"] * scale,
        "avg_pace_s_per_km": run_features["avg_pace_s_per_km"],
        "avg_heart_rate_bpm": run_features["avg_heart_rate_bpm"],
        "temperature_c": run_features["temperature_c"],
        "time_of_day_hour": run_features["time_of_day_hour"],
        "fatigue_score": run_features["fatigue_score"],
    }
    return predict(model, features)
