"""
map_visualizer.py

Responsible for:
- Folium maps
- Route rendering
- Colour scaling
- Pace heatmaps
"""

import folium
import folium.plugins
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Pace thresholds in seconds/km used for colour scaling
PACE_SLOW_THRESHOLD = 420.0   # 7:00 /km  → blue
PACE_FAST_THRESHOLD = 240.0   # 4:00 /km  → red


def pace_to_colour(
    pace_seconds_per_km: float,
    slow_threshold: float = PACE_SLOW_THRESHOLD,
    fast_threshold: float = PACE_FAST_THRESHOLD,
) -> str:
    """
    Map a pace value to a hex colour string.

    Colour scale:
        Blue  → slower pace  (>= slow_threshold)
        Green → moderate pace
        Red   → faster pace  (<= fast_threshold)

    Args:
        pace_seconds_per_km: Pace in seconds per kilometre.
        slow_threshold: Pace (s/km) considered slow.
        fast_threshold: Pace (s/km) considered fast.

    Returns:
        A hex colour string (e.g. "#FF0000").
    """
    if pace_seconds_per_km <= 0:
        return "#808080"  # grey for invalid/zero pace

    # Normalise pace to [0, 1] where 0 = fast, 1 = slow
    ratio = (pace_seconds_per_km - fast_threshold) / (
        slow_threshold - fast_threshold
    )
    ratio = max(0.0, min(1.0, ratio))

    # Interpolate: fast (red) → moderate (green) → slow (blue)
    if ratio < 0.5:
        # Red → Green
        t = ratio / 0.5
        r = int(255 * (1 - t))
        g = int(255 * t)
        b = 0
    else:
        # Green → Blue
        t = (ratio - 0.5) / 0.5
        r = 0
        g = int(255 * (1 - t))
        b = int(255 * t)

    return f"#{r:02X}{g:02X}{b:02X}"


def create_base_map(
    coordinates: List[tuple[float, float]],
    zoom_start: int = 14,
) -> folium.Map:
    """
    Create a Folium map auto-centred on the provided coordinates.

    Args:
        coordinates: List of (latitude, longitude) tuples.
        zoom_start: Initial zoom level.

    Returns:
        A Folium Map object.

    Raises:
        ValueError: If coordinates list is empty.
    """
    if not coordinates:
        raise ValueError("Cannot create a map with no coordinates.")

    avg_lat = sum(c[0] for c in coordinates) / len(coordinates)
    avg_lon = sum(c[1] for c in coordinates) / len(coordinates)

    return folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )


def render_pace_route(
    fmap: folium.Map,
    coordinates: List[tuple[float, float]],
    paces: List[float],
    activity_name: str = "Run",
) -> folium.Map:
    """
    Render a route onto a Folium map, colouring each segment by pace.

    Args:
        fmap: An existing Folium Map to draw onto.
        coordinates: List of (latitude, longitude) tuples.
        paces: Pace values (s/km) per segment. Length = len(coordinates) - 1.
        activity_name: Label used in tooltips.

    Returns:
        The updated Folium Map.
    """
    if len(coordinates) < 2:
        logger.warning("Not enough coordinates to draw a route.")
        return fmap

    if len(paces) != len(coordinates) - 1:
        raise ValueError(
            f"Expected {len(coordinates) - 1} pace values, got {len(paces)}."
        )

    for i, pace in enumerate(paces):
        colour = pace_to_colour(pace)
        segment = [coordinates[i], coordinates[i + 1]]

        minutes, seconds = divmod(int(pace), 60)
        tooltip_text = (
            f"{activity_name} · Segment {i + 1}<br>"
            f"Pace: {minutes}:{seconds:02d} /km"
        )

        folium.PolyLine(
            locations=segment,
            color=colour,
            weight=4,
            opacity=0.85,
            tooltip=folium.Tooltip(tooltip_text),
        ).add_to(fmap)

    logger.info(f"Rendered {len(paces)} segments for '{activity_name}'.")
    return fmap


def add_start_end_markers(
    fmap: folium.Map,
    coordinates: List[tuple[float, float]],
    activity_name: str = "Run",
) -> folium.Map:
    """
    Add start (green) and end (red) markers to the map.

    Args:
        fmap: Folium Map object.
        coordinates: List of (latitude, longitude) tuples.
        activity_name: Used in marker popups.

    Returns:
        Updated Folium Map.
    """
    if not coordinates:
        return fmap

    folium.Marker(
        location=list(coordinates[0]),
        popup=f"{activity_name} – Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fmap)

    folium.Marker(
        location=list(coordinates[-1]),
        popup=f"{activity_name} – Finish",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fmap)

    return fmap


def build_activity_map(
    coordinates: List[tuple[float, float]],
    paces: List[float],
    activity_name: str = "Run",
    zoom_start: int = 14,
) -> folium.Map:
    """
    Build a complete pace-coloured activity map.

    Convenience wrapper that creates the base map, renders the route,
    and adds start/end markers.

    Args:
        coordinates: List of (latitude, longitude) tuples.
        paces: Pace values (s/km) per segment.
        activity_name: Display name for the activity.
        zoom_start: Initial zoom level.

    Returns:
        A fully rendered Folium Map.
    """
    fmap = create_base_map(coordinates, zoom_start=zoom_start)
    fmap = render_pace_route(fmap, coordinates, paces, activity_name)
    fmap = add_start_end_markers(fmap, coordinates, activity_name)
    return fmap


def export_map_html(fmap: folium.Map, output_path: str) -> None:
    """
    Export a Folium map to an HTML file.

    Args:
        fmap: The Folium Map to export.
        output_path: Destination file path (e.g. "output/run_map.html").
    """
    fmap.save(output_path)
    logger.info(f"Map exported to: {output_path}")


# --- Example usage ---
if __name__ == "__main__":
    sample_coords = [
        (51.5074, -0.1278),
        (51.5080, -0.1265),
        (51.5090, -0.1250),
        (51.5100, -0.1235),
    ]
    # Paces in s/km: fast → slow → moderate
    sample_paces = [270.0, 390.0, 330.0]

    activity_map = build_activity_map(sample_coords, sample_paces, "Sample Run")
    export_map_html(activity_map, "sample_run_map.html")
    print("Map saved to sample_run_map.html")
