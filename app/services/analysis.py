"""
Analysis service — wraps existing src/ modules for the web API.
"""

import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure src/ is importable (same pattern as CLI scripts)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from parser import (  # noqa: E402
    extract_coordinates,
    extract_elevations,
    extract_timestamps,
    extract_track_points,
    load_gpx_file,
)
from pace_calculator import (  # noqa: E402
    calculate_cumulative_distance,
    calculate_segment_distances,
    calculate_segment_paces,
    smooth_gps_coordinates,
)
from utils import (  # noqa: E402
    deduplicate_timestamps,
    elevation_gain,
    elapsed_seconds,
    format_distance,
    format_duration,
    format_pace,
)
from map_visualizer import build_activity_map, export_map_html  # noqa: E402


def analyze_run(
    gpx_filepath: str,
    output_dir: str = "output",
    smoothing_window: int = 3,
    user_location: tuple[float, float] | None = None,
    map_filename_suffix: str = "",
    skip_chart: bool = False,
) -> dict[str, Any]:
    """
    Analyse a GPX run and generate visualisations.

    Returns structured stats plus URLs to generated assets.
    """
    os.makedirs(output_dir, exist_ok=True)

    run_name = os.path.splitext(os.path.basename(gpx_filepath))[0]

    gpx = load_gpx_file(gpx_filepath)
    track_points = extract_track_points(gpx)
    coordinates = extract_coordinates(track_points)
    timestamps = extract_timestamps(track_points)
    elevations = extract_elevations(track_points)

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

    fastest_pace = min(valid_paces) if valid_paces else None
    slowest_pace = max(valid_paces) if valid_paces else None

    chart_filename = f"{run_name}_pace_chart.png"
    map_filename = f"{run_name}{map_filename_suffix}_pace_map.html"

    if not skip_chart:
        _generate_pace_chart(
            cumulative=cumulative,
            segment_paces=segment_paces,
            avg_pace=avg_pace,
            run_name=run_name,
            output_path=os.path.join(output_dir, chart_filename),
        )

    # Use raw GPS coordinates for the map so the full route shape is preserved
    map_paces = calculate_segment_paces(coordinates, timestamps_clean)

    activity_map = build_activity_map(
        coordinates=coordinates,
        paces=map_paces,
        activity_name=run_name,
        zoom_start=14,
        user_location=user_location,
        fit_route=True,
    )
    export_map_html(activity_map, os.path.join(output_dir, map_filename))

    return {
        "run_name": run_name,
        "stats": {
            "distance": format_distance(total_distance_m),
            "distance_m": total_distance_m,
            "duration": format_duration(total_elapsed or 0),
            "duration_s": total_elapsed or 0,
            "avg_pace": format_pace(avg_pace),
            "avg_pace_s_per_km": avg_pace,
            "elevation_gain_m": round(elev_gain, 1),
            "gps_points": len(track_points),
            "segments": len(segment_paces),
            "fastest_pace": format_pace(fastest_pace) if fastest_pace else None,
            "slowest_pace": format_pace(slowest_pace) if slowest_pace else None,
            "start_time": timestamps[0].isoformat() if timestamps[0] else None,
            "end_time": timestamps[-1].isoformat() if timestamps[-1] else None,
        },
        "outputs": {
            "pace_chart": None if skip_chart else f"/output/{chart_filename}",
            "pace_map": f"/output/{map_filename}",
        },
    }


def _generate_pace_chart(
    cumulative: list[float],
    segment_paces: list[float],
    avg_pace: float,
    run_name: str,
    output_path: str,
) -> None:
    """Generate and save a pace-over-distance chart."""
    cum_km = [d / 1000 for d in cumulative[:-1]]
    pace_mins = [p / 60 for p in segment_paces]
    filtered = [(k, p) for k, p in zip(cum_km, pace_mins) if 0 < p < 20]
    x_vals, y_vals = zip(*filtered) if filtered else ([], [])

    fig, ax = plt.subplots(figsize=(14, 5))
    strava_orange = "#FC4C02"
    ax.plot(x_vals, y_vals, linewidth=1.5, color=strava_orange, alpha=0.85, label="Pace")
    ax.fill_between(x_vals, y_vals, alpha=0.15, color=strava_orange)
    ax.axhline(
        avg_pace / 60,
        color="#D44200",
        linestyle="--",
        linewidth=2,
        label=f"Avg {format_pace(avg_pace)}",
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v)}:{int((v % 1) * 60):02d}")
    )
    ax.invert_yaxis()
    ax.set_xlabel("Distance (km)", fontsize=12)
    ax.set_ylabel("Pace (min/km)", fontsize=12)
    ax.set_title(f"Pace Over Distance - {run_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
