"""
Generate new running route candidates near a location using OSRM foot routing.
"""

import json
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

OSRM_BASE = "https://router.project-osrm.org/route/v1/foot"
EARTH_RADIUS_M = 6_371_000
USER_AGENT = "PaceMap-AI/2.0 (running route generator)"
MAX_ROUTE_POINTS = 150


def destination_point(
    lat: float, lon: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """Return (lat, lon) reached by traveling distance_m along bearing_deg."""
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    bearing_r = math.radians(bearing_deg)
    angular = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat_r) * math.cos(angular)
        + math.cos(lat_r) * math.sin(angular) * math.cos(bearing_r)
    )
    lon2 = lon_r + math.atan2(
        math.sin(bearing_r) * math.sin(angular) * math.cos(lat_r),
        math.cos(angular) - math.sin(lat_r) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _downsample_coordinates(
    coordinates: list[tuple[float, float]], max_points: int = MAX_ROUTE_POINTS
) -> list[tuple[float, float]]:
    if len(coordinates) <= max_points:
        return coordinates
    step = len(coordinates) / max_points
    indices = sorted({int(i * step) for i in range(max_points)} | {len(coordinates) - 1})
    return [coordinates[i] for i in indices]


def _fetch_osrm_route(waypoints: list[tuple[float, float]]) -> Optional[dict[str, Any]]:
    """Request a foot route through OSRM (simplified geometry for speed)."""
    if len(waypoints) < 2:
        return None

    coord_str = ";".join(f"{lon},{lat}" for lat, lon in waypoints)
    url = (
        f"{OSRM_BASE}/{coord_str}"
        "?overview=simplified&geometries=geojson&steps=false&alternatives=false"
    )

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    if data.get("code") != "Ok" or not data.get("routes"):
        return None

    route = data["routes"][0]
    geometry = route["geometry"]
    if geometry.get("type") != "LineString":
        return None

    coordinates = _downsample_coordinates(
        [(pt[1], pt[0]) for pt in geometry["coordinates"]]
    )
    if len(coordinates) < 2:
        return None

    return {
        "coordinates": coordinates,
        "distance_m": route["distance"],
        "distance_km": route["distance"] / 1000,
    }


def _loop_waypoints(
    start_lat: float, start_lon: float, bearing_deg: float, leg_km: float
) -> list[tuple[float, float]]:
    leg_m = leg_km * 1000
    wp1 = destination_point(start_lat, start_lon, bearing_deg, leg_m)
    wp2 = destination_point(start_lat, start_lon, bearing_deg + 120, leg_m)
    return [(start_lat, start_lon), wp1, wp2, (start_lat, start_lon)]


def _out_and_back_waypoints(
    start_lat: float, start_lon: float, bearing_deg: float, half_km: float
) -> list[tuple[float, float]]:
    half_m = half_km * 1000
    turnaround = destination_point(start_lat, start_lon, bearing_deg, half_m)
    return [(start_lat, start_lon), turnaround, (start_lat, start_lon)]


def generate_route_candidates(
    start_lat: float,
    start_lon: float,
    target_distance_km: float,
) -> list[dict[str, Any]]:
    """
    Generate route candidates in parallel via OSRM foot routing.

    Tries loop and out-and-back shapes in 8 compass directions.
    """
    bearings = [0, 45, 90, 135, 180, 225, 270, 315]
    direction_names = {
        0: "North", 45: "Northeast", 90: "East", 135: "Southeast",
        180: "South", 225: "Southwest", 270: "West", 315: "Northwest",
    }

    road_factor = 0.65
    loop_leg_km = (target_distance_km / 3) * road_factor
    half_km = (target_distance_km / 2) * road_factor

    jobs: list[tuple[str, int, str, list[tuple[float, float]]]] = []
    for bearing in bearings:
        jobs.append((
            "loop", bearing, direction_names[bearing],
            _loop_waypoints(start_lat, start_lon, bearing, loop_leg_km),
        ))
        jobs.append((
            "out-and-back", bearing, direction_names[bearing],
            _out_and_back_waypoints(start_lat, start_lon, bearing, half_km),
        ))

    candidates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(_fetch_osrm_route, waypoints): (shape, bearing, direction)
            for shape, bearing, direction, waypoints in jobs
        }
        for future in as_completed(future_map):
            shape_name, bearing, direction = future_map[future]
            route = future.result()
            if not route:
                continue

            dist_km = route["distance_km"]
            if dist_km < target_distance_km * 0.45 or dist_km > target_distance_km * 1.55:
                continue

            candidates.append({
                "name": f"{shape_name.title()} — {direction}",
                "shape": shape_name,
                "bearing_deg": bearing,
                "direction": direction,
                "coordinates": route["coordinates"],
                "distance_km": round(dist_km, 2),
                "distance_m": route["distance_m"],
                "start_lat": start_lat,
                "start_lon": start_lon,
            })

    candidates.sort(key=lambda c: abs(c["distance_km"] - target_distance_km))
    return candidates
