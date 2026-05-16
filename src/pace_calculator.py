"""
pace_calculator.py

Responsible for:
- Distance calculations
- Speed calculations
- Pace calculations
- Moving averages
- Smoothing noisy GPS data
"""

import math
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000  # metres


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate the great-circle distance between two GPS coordinates using
    the Haversine formula.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in metres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_segment_distances(
    coordinates: List[tuple[float, float]],
) -> List[float]:
    """
    Calculate the distance (metres) between each consecutive pair of coordinates.

    Args:
        coordinates: List of (latitude, longitude) tuples.

    Returns:
        List of distances in metres. Length is len(coordinates) - 1.
    """
    if len(coordinates) < 2:
        return []

    return [
        haversine_distance(
            coordinates[i][0], coordinates[i][1],
            coordinates[i + 1][0], coordinates[i + 1][1],
        )
        for i in range(len(coordinates) - 1)
    ]


def calculate_cumulative_distance(segment_distances: List[float]) -> List[float]:
    """
    Calculate cumulative distance at each point.

    Args:
        segment_distances: List of per-segment distances in metres.

    Returns:
        Cumulative distance list starting at 0.0.
    """
    cumulative = [0.0]
    for d in segment_distances:
        cumulative.append(cumulative[-1] + d)
    return cumulative


def calculate_speed(
    distance_m: float,
    time_seconds: float,
) -> float:
    """
    Calculate speed in metres per second.

    Args:
        distance_m: Distance in metres.
        time_seconds: Time elapsed in seconds.

    Returns:
        Speed in m/s, or 0.0 if time_seconds <= 0 (prevents divide-by-zero).
    """
    if time_seconds <= 0:
        logger.debug("time_seconds <= 0; returning speed of 0.0.")
        return 0.0
    return distance_m / time_seconds


def calculate_pace(
    distance_m: float,
    time_seconds: float,
) -> float:
    """
    Calculate running pace in seconds per kilometre.

    Args:
        distance_m: Distance in metres.
        time_seconds: Time elapsed in seconds.

    Returns:
        Pace in seconds/km, or 0.0 if distance_m <= 0.
    """
    if distance_m <= 0:
        logger.debug("distance_m <= 0; returning pace of 0.0.")
        return 0.0
    return (time_seconds / distance_m) * 1000.0


def calculate_segment_paces(
    coordinates: List[tuple[float, float]],
    timestamps: List[Optional[datetime]],
) -> List[float]:
    """
    Calculate pace (seconds/km) for each segment between consecutive points.

    Skips segments with missing, duplicate, or out-of-order timestamps.

    Args:
        coordinates: List of (latitude, longitude) tuples.
        timestamps: Corresponding list of datetime objects (may contain None).

    Returns:
        List of pace values in seconds/km. Segments with invalid data get 0.0.
    """
    if len(coordinates) != len(timestamps):
        raise ValueError("coordinates and timestamps must have the same length.")

    segment_distances = calculate_segment_distances(coordinates)
    paces: List[float] = []

    for i, dist in enumerate(segment_distances):
        t_start = timestamps[i]
        t_end = timestamps[i + 1]

        if t_start is None or t_end is None:
            logger.warning(f"Missing timestamp at segment {i}; pace set to 0.0.")
            paces.append(0.0)
            continue

        elapsed = (t_end - t_start).total_seconds()

        if elapsed <= 0:
            logger.warning(
                f"Non-positive elapsed time at segment {i} "
                f"({elapsed}s); pace set to 0.0."
            )
            paces.append(0.0)
            continue

        paces.append(calculate_pace(dist, elapsed))

    return paces


def moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    Apply a simple moving average to smooth a list of values.

    Args:
        values: Input list of floats.
        window: Number of points to average over. Must be >= 1.

    Returns:
        Smoothed list of the same length. Edge values use smaller windows.
    """
    if window < 1:
        raise ValueError("window must be >= 1.")

    smoothed: List[float] = []
    n = len(values)

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        segment = values[start:end]
        smoothed.append(sum(segment) / len(segment))

    return smoothed


def smooth_gps_coordinates(
    coordinates: List[tuple[float, float]],
    window: int = 3,
) -> List[tuple[float, float]]:
    """
    Smooth noisy GPS coordinates using a moving average on lat/lon separately.

    Args:
        coordinates: List of (latitude, longitude) tuples.
        window: Smoothing window size.

    Returns:
        Smoothed list of (latitude, longitude) tuples.
    """
    if not coordinates:
        return []

    latitudes = [c[0] for c in coordinates]
    longitudes = [c[1] for c in coordinates]

    smoothed_lats = moving_average(latitudes, window)
    smoothed_lons = moving_average(longitudes, window)

    return list(zip(smoothed_lats, smoothed_lons))


# --- Example usage ---
if __name__ == "__main__":
    sample_coords = [
        (51.5074, -0.1278),
        (51.5080, -0.1265),
        (51.5090, -0.1250),
    ]
    from datetime import timedelta

    base_time = datetime(2024, 1, 1, 8, 0, 0)
    sample_times = [base_time + timedelta(seconds=i * 30) for i in range(3)]

    distances = calculate_segment_distances(sample_coords)
    paces = calculate_segment_paces(sample_coords, sample_times)
    cumulative = calculate_cumulative_distance(distances)

    print(f"Segment distances (m) : {[round(d, 2) for d in distances]}")
    print(f"Cumulative distance   : {[round(d, 2) for d in cumulative]}")
    print(f"Segment paces (s/km)  : {[round(p, 1) for p in paces]}")
    print(f"Smoothed coords       : {smooth_gps_coordinates(sample_coords)}")
