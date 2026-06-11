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

# Pace thresholds in seconds/km used for colour scaling (Strava orange palette)
PACE_SLOW_THRESHOLD = 420.0   # 7:00 /km  → light orange
PACE_FAST_THRESHOLD = 240.0   # 4:00 /km  → Strava orange
STRAVA_ORANGE = "#FC4C02"
STRAVA_ORANGE_LIGHT = "#FFC299"
MAX_PACE_SEGMENTS = 600       # downsample pace overlay above this


def pace_to_colour(
    pace_seconds_per_km: float,
    slow_threshold: float = PACE_SLOW_THRESHOLD,
    fast_threshold: float = PACE_FAST_THRESHOLD,
) -> str:
    """
    Map a pace value to a hex colour string.

    Colour scale (Strava-inspired orange gradient):
        Light orange → slower pace  (>= slow_threshold)
        Mid orange   → moderate pace
        Strava orange → faster pace  (<= fast_threshold)

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

    # Interpolate: fast (#FC4C02) → mid (#FF8C55) → slow (#FFC299)
    fast_rgb = (252, 76, 2)
    mid_rgb = (255, 140, 85)
    slow_rgb = (255, 194, 153)

    if ratio < 0.5:
        t = ratio / 0.5
        r = int(fast_rgb[0] + (mid_rgb[0] - fast_rgb[0]) * t)
        g = int(fast_rgb[1] + (mid_rgb[1] - fast_rgb[1]) * t)
        b = int(fast_rgb[2] + (mid_rgb[2] - fast_rgb[2]) * t)
    else:
        t = (ratio - 0.5) / 0.5
        r = int(mid_rgb[0] + (slow_rgb[0] - mid_rgb[0]) * t)
        g = int(mid_rgb[1] + (slow_rgb[1] - mid_rgb[1]) * t)
        b = int(mid_rgb[2] + (slow_rgb[2] - mid_rgb[2]) * t)

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


def downsample_route(
    coordinates: List[tuple[float, float]],
    paces: List[float],
    max_segments: int = MAX_PACE_SEGMENTS,
) -> tuple[List[tuple[float, float]], List[float]]:
    """
    Downsample a route for pace-coloured overlay while preserving endpoints.

    The full GPS trace is drawn separately; this only thins pace segments.
    """
    if len(paces) <= max_segments:
        return coordinates, paces

    indices = sorted({
        int(i * len(paces) / max_segments) for i in range(max_segments)
    } | {len(paces) - 1})

    new_coords = [coordinates[0]]
    new_paces = []
    for idx in indices:
        new_paces.append(paces[idx])
        new_coords.append(coordinates[idx + 1])

    return new_coords, new_paces


def add_full_route_trace(
    fmap: folium.Map,
    coordinates: List[tuple[float, float]],
    color: str = STRAVA_ORANGE,
    weight: int = 5,
    opacity: float = 0.9,
) -> folium.Map:
    """Draw the complete GPS track as a single continuous polyline."""
    if len(coordinates) < 2:
        return fmap

    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in coordinates],
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip=folium.Tooltip("Full GPS track"),
    ).add_to(fmap)

    logger.info(f"Drew full route trace with {len(coordinates)} points.")
    return fmap


def fit_map_to_route(
    fmap: folium.Map,
    coordinates: List[tuple[float, float]],
    user_location: Optional[tuple[float, float]] = None,
    padding: tuple[int, int] = (40, 40),
) -> folium.Map:
    """Zoom the map to fit the entire route (and optional user location)."""
    if not coordinates:
        return fmap

    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]

    if user_location:
        lats.append(user_location[0])
        lons.append(user_location[1])

    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    fmap.fit_bounds(bounds, padding=padding)
    return fmap


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
    user_location: Optional[tuple[float, float]] = None,
    fit_route: bool = True,
) -> folium.Map:
    """
    Build a complete pace-coloured activity map.

    Draws the full GPS track first, then overlays pace-coloured segments,
    and auto-fits the viewport to the route shape.

    Args:
        coordinates: List of (latitude, longitude) tuples.
        paces: Pace values (s/km) per segment.
        activity_name: Display name for the activity.
        zoom_start: Initial zoom level (used before fit_bounds).
        user_location: Optional (lat, lon) to mark the user's position.
        fit_route: Whether to auto-zoom to the full route.

    Returns:
        A fully rendered Folium Map.
    """
    fmap = create_base_map(coordinates, zoom_start=zoom_start)
    fmap = add_full_route_trace(fmap, coordinates)

    display_coords, display_paces = downsample_route(coordinates, paces)
    if len(display_paces) == len(display_coords) - 1:
        fmap = render_pace_route(fmap, display_coords, display_paces, activity_name)

    fmap = add_start_end_markers(fmap, coordinates, activity_name)

    if user_location:
        folium.Marker(
            location=list(user_location),
            popup="Your location",
            icon=folium.Icon(color="orange", icon="info-sign"),
        ).add_to(fmap)

    if fit_route:
        fmap = fit_map_to_route(fmap, coordinates, user_location=user_location)

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
