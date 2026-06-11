"""
Fetch elevation data for route coordinates.
"""

import json
import urllib.request
from typing import Optional

from pacemap.utils import elevation_gain

USER_AGENT = "PaceMap-AI/2.0 (elevation lookup)"
BATCH_SIZE = 100


def _sample_coordinates(
    coordinates: list[tuple[float, float]], max_points: int = 30
) -> list[tuple[float, float]]:
    if len(coordinates) <= max_points:
        return coordinates
    step = len(coordinates) / max_points
    indices = sorted({int(i * step) for i in range(max_points)} | {len(coordinates) - 1})
    return [coordinates[i] for i in indices]


def fetch_elevations(
    coordinates: list[tuple[float, float]], max_points: int = 30
) -> list[Optional[float]]:
    """Look up elevations for coordinates using Open Topo Data."""
    sampled = _sample_coordinates(coordinates, max_points=max_points)
    elevations: list[Optional[float]] = []

    for i in range(0, len(sampled), BATCH_SIZE):
        batch = sampled[i : i + BATCH_SIZE]
        loc_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
        url = f"https://api.opentopodata.org/v1/aster30m?locations={loc_str}"

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            elevations.extend([None] * len(batch))
            continue

        for result in data.get("results", []):
            elev = result.get("elevation")
            elevations.append(float(elev) if elev is not None else None)

    return elevations


def calculate_route_elevation(
    coordinates: list[tuple[float, float]], max_points: int = 30
) -> float:
    """Return total elevation gain (m) for a route."""
    if len(coordinates) < 2:
        return 0.0
    elevations = fetch_elevations(coordinates, max_points=max_points)
    return elevation_gain(elevations)
