"""
ML-based route recommender — generates NEW routes near a location.
"""

import math
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pace_calculator import haversine_distance  # noqa: E402
from utils import format_distance, format_duration, format_pace  # noqa: E402

from app.services.analysis import analyze_run  # noqa: E402
from app.services.elevation_service import calculate_route_elevation  # noqa: E402
from app.services.gpx_builder import write_route_gpx_with_distance  # noqa: E402
from app.services.route_generator import generate_route_candidates  # noqa: E402
from app.services.run_profile import load_model, predict_finish_time  # noqa: E402


WEIGHTS = {
    "distance": 0.40,
    "elevation": 0.25,
    "novelty": 0.20,
    "profile_fit": 0.15,
}


def _gaussian_score(value: float, target: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if abs(value - target) < 0.01 else 0.0
    return math.exp(-0.5 * ((value - target) / sigma) ** 2)


def _distance_score(route_distance_km: float, target_km: float) -> float:
    tolerance = max(target_km * 0.2, 0.8)
    return _gaussian_score(route_distance_km, target_km, tolerance)


def _elevation_score(
    route_elevation: float,
    profile_avg: float,
    profile_std: float,
) -> float:
    sigma = profile_std if profile_std > 0 else max(profile_avg * 0.5, 20)
    return _gaussian_score(route_elevation, profile_avg, sigma)


def _novelty_score(
    candidate_coords: list[tuple[float, float]],
    historical_runs: list[dict[str, Any]],
) -> float:
    """
    Fast novelty check using run midpoints from learned data (no GPX reload).
    """
    if not historical_runs:
        return 1.0

    mid_idx = len(candidate_coords) // 2
    mid_lat, mid_lon = candidate_coords[mid_idx]

    nearest_m = float("inf")
    for run in historical_runs:
        rlat = run.get("start_lat")
        rlon = run.get("start_lon")
        if rlat is None or rlon is None:
            continue
        nearest_m = min(nearest_m, haversine_distance(mid_lat, mid_lon, rlat, rlon))

    if nearest_m == float("inf"):
        return 1.0
    if nearest_m < 200:
        return 0.5
    if nearest_m < 500:
        return 0.75
    return 1.0


def _profile_fit_score(
    route_features: dict[str, Any],
    profile: dict[str, Any],
    model,
) -> float:
    """Score how well the route suits the runner's ability."""
    target_km = route_features["distance_km"]
    if model:
        predicted_s = predict_finish_time(model, route_features, target_km)
        predicted_pace = predicted_s / target_km if target_km > 0 else 0
    else:
        predicted_pace = profile["avg_pace_s_per_km"]

    typical_pace = profile["avg_pace_s_per_km"]
    pace_diff = abs(predicted_pace - typical_pace)
    tolerance = max(typical_pace * 0.2, 45)
    return _gaussian_score(pace_diff, 0, tolerance)


def _score_single_candidate(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    model,
    target_distance_km: float,
    historical_runs: list[dict[str, Any]],
    elev_gain: float,
) -> dict[str, Any]:
    route_features = {
        "distance_km": candidate["distance_km"],
        "elevation_gain_m": elev_gain,
        "avg_pace_s_per_km": profile["avg_pace_s_per_km"],
        "avg_heart_rate_bpm": profile.get("avg_heart_rate_bpm", 155),
        "temperature_c": 20.0,
        "time_of_day_hour": 8,
        "fatigue_score": 5.0,
    }

    dist_score = _distance_score(candidate["distance_km"], target_distance_km)
    elev_score = _elevation_score(
        elev_gain,
        profile["avg_elevation_gain_m"],
        profile.get("elevation_std_m", 0) or 30,
    )
    novelty_score = _novelty_score(candidate["coordinates"], historical_runs)
    fit_score = _profile_fit_score(route_features, profile, model)

    total = (
        WEIGHTS["distance"] * dist_score
        + WEIGHTS["elevation"] * elev_score
        + WEIGHTS["novelty"] * novelty_score
        + WEIGHTS["profile_fit"] * fit_score
    )

    predicted_time_s = None
    if model:
        predicted_time_s = predict_finish_time(model, route_features, candidate["distance_km"])

    return {
        **candidate,
        "elevation_gain_m": round(elev_gain, 1),
        "avg_pace": format_pace(profile["avg_pace_s_per_km"]),
        "predicted_finish_time": format_duration(predicted_time_s) if predicted_time_s else None,
        "predicted_finish_time_s": predicted_time_s,
        "scores": {
            "distance": round(dist_score, 3),
            "elevation": round(elev_score, 3),
            "novelty": round(novelty_score, 3),
            "profile_fit": round(fit_score, 3),
            "total": round(total, 3),
        },
    }


def score_candidates(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    model,
    target_distance_km: float,
    historical_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score generated route candidates and return sorted list."""
    prelim = []
    for candidate in candidates:
        dist = _distance_score(candidate["distance_km"], target_distance_km)
        novelty = _novelty_score(candidate["coordinates"], historical_runs)
        prelim.append((dist * 0.7 + novelty * 0.3, candidate))

    prelim.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [c for _, c in prelim[:4]]

    scored = []
    for i, candidate in enumerate(top_candidates):
        # Only fetch elevation for top 2 — use profile average for the rest
        if i < 2:
            elev_gain = calculate_route_elevation(candidate["coordinates"], max_points=20)
        else:
            elev_gain = profile["avg_elevation_gain_m"]

        scored.append(
            _score_single_candidate(
                candidate, profile, model, target_distance_km, historical_runs, elev_gain
            )
        )

    scored.sort(key=lambda r: r["scores"]["total"], reverse=True)
    return scored


def _build_reasoning(
    best: dict[str, Any],
    profile: dict[str, Any],
    target_distance_km: float,
) -> list[str]:
    reasons = [
        f"New {best['shape']} route heading {best['direction']} — not a run from your history.",
        f"Follows footpaths and trails via OpenStreetMap ({best['distance_km']} km).",
    ]

    dist_diff = abs(best["distance_km"] - target_distance_km)
    if dist_diff < 1:
        reasons.append(f"Distance closely matches your {target_distance_km} km target.")
    else:
        reasons.append(f"Closest generated match to your {target_distance_km} km goal.")

    if best["scores"]["novelty"] >= 0.7:
        reasons.append("This path is significantly different from your past runs.")
    elif best["scores"]["novelty"] < 0.4:
        reasons.append("Route overlaps your usual area — try another direction for more variety.")

    if best["scores"]["elevation"] > 0.6:
        reasons.append(
            f"Elevation gain ({best['elevation_gain_m']} m) suits your typical "
            f"range (avg {profile['avg_elevation_gain_m']} m)."
        )
    elif best["elevation_gain_m"] < profile["avg_elevation_gain_m"] * 0.5:
        reasons.append("Relatively flat route — good for speed or recovery.")
    else:
        reasons.append("Includes more climbing than your average — a solid training challenge.")

    if best.get("predicted_finish_time"):
        reasons.append(f"Estimated finish time: {best['predicted_finish_time']} based on your profile.")

    if profile.get("model_trained"):
        reasons.append(f"Scored using your ML model (R²={profile['model_metrics']['r2']:.2f}).")

    return reasons


def _route_to_response(
    route: dict[str, Any],
    route_id: str,
    outputs: dict[str, Any],
    reasoning: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Format a scored route for the API response."""
    payload = {
        "route_id": route_id,
        "name": route["name"],
        "shape": route["shape"],
        "direction": route["direction"],
        "distance_km": route["distance_km"],
        "distance": format_distance(route["distance_km"] * 1000),
        "elevation_gain_m": route["elevation_gain_m"],
        "avg_pace": route["avg_pace"],
        "predicted_finish_time": route.get("predicted_finish_time"),
        "predicted_finish_time_s": route.get("predicted_finish_time_s"),
        "scores": route["scores"],
        "outputs": outputs,
        "is_new_route": True,
    }
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def _build_route_map(
    route: dict[str, Any],
    profile: dict[str, Any],
    gpx_dir: Path,
    output_dir: Path,
    target_lat: float,
    target_lon: float,
    smoothing_window: int,
    map_suffix: str,
) -> tuple[str, dict[str, Any]]:
    """Write GPX and generate map HTML for a route candidate."""
    gpx_dir.mkdir(parents=True, exist_ok=True)
    route_id = uuid.uuid4().hex[:8]
    gpx_path = gpx_dir / f"new_route_{route_id}.gpx"

    write_route_gpx_with_distance(
        coordinates=route["coordinates"],
        output_path=str(gpx_path),
        route_name=route["name"],
        distance_km=route["distance_km"],
        pace_s_per_km=profile["avg_pace_s_per_km"],
        elevations=None,
    )

    analysis = analyze_run(
        gpx_filepath=str(gpx_path),
        output_dir=str(output_dir),
        smoothing_window=smoothing_window,
        user_location=(target_lat, target_lon),
        map_filename_suffix=map_suffix,
        skip_chart=True,
    )
    return route_id, analysis["outputs"]


def recommend_route(
    profile: dict[str, Any],
    models_dir: Path,
    output_dir: Path,
    gpx_dir: Path,
    target_lat: float,
    target_lon: float,
    target_distance_km: float,
    historical_runs: Optional[list[dict[str, Any]]] = None,
    smoothing_window: int = 3,
) -> dict[str, Any]:
    """Generate and return the best NEW route recommendation."""
    candidates = generate_route_candidates(target_lat, target_lon, target_distance_km)

    if not candidates:
        raise ValueError(
            "Could not generate routes near this location. "
            "Try a different area or a distance between 3–21 km."
        )

    model = load_model(models_dir) if profile.get("model_trained") else None
    scored = score_candidates(
        candidates, profile, model, target_distance_km, historical_runs or []
    )

    novel = [r for r in scored if r["scores"]["novelty"] >= 0.4]
    best = novel[0] if novel else scored[0]
    best_reasoning = _build_reasoning(best, profile, target_distance_km)

    best_id, best_outputs = _build_route_map(
        best, profile, gpx_dir, output_dir, target_lat, target_lon,
        smoothing_window, "_new",
    )
    recommended = _route_to_response(best, best_id, best_outputs, reasoning=best_reasoning)

    alternatives = []
    for alt in scored:
        if alt["name"] == best["name"]:
            continue
        if len(alternatives) >= 3:
            break
        alt_id, alt_outputs = _build_route_map(
            alt, profile, gpx_dir, output_dir, target_lat, target_lon,
            smoothing_window, f"_alt_{uuid.uuid4().hex[:8]}",
        )
        alternatives.append(_route_to_response(alt, alt_id, alt_outputs))

    return {
        "recommended": recommended,
        "alternatives": alternatives,
        "target": {
            "latitude": target_lat,
            "longitude": target_lon,
            "distance_km": target_distance_km,
        },
        "profile_summary": {
            "total_runs": profile["total_runs"],
            "avg_pace": profile["avg_pace"],
            "avg_distance_km": profile["avg_distance_km"],
            "model_trained": profile.get("model_trained", False),
        },
        "outputs": best_outputs,
    }
