"""
utils.py

Responsible for:
- Helper functions
- Reusable calculations
- Formatting utilities
"""

from datetime import datetime, timedelta
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def format_pace(pace_seconds_per_km: float) -> str:
    """
    Format a pace value (seconds/km) as a human-readable MM:SS string.

    Args:
        pace_seconds_per_km: Pace in seconds per kilometre.

    Returns:
        String in the format "M:SS /km" (e.g. "5:30 /km").
        Returns "--:-- /km" for zero or negative values.
    """
    if pace_seconds_per_km <= 0:
        return "--:-- /km"
    minutes, seconds = divmod(int(pace_seconds_per_km), 60)
    return f"{minutes}:{seconds:02d} /km"


def format_duration(total_seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable HH:MM:SS string.

    Args:
        total_seconds: Duration in seconds.

    Returns:
        String in the format "H:MM:SS" (e.g. "1:23:45").
    """
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_distance(distance_m: float) -> str:
    """
    Format a distance in metres to a human-readable string.

    Displays in km if >= 1000 m, otherwise in metres.

    Args:
        distance_m: Distance in metres.

    Returns:
        Formatted string (e.g. "10.5 km" or "850 m").
    """
    if distance_m >= 1000:
        return f"{distance_m / 1000:.2f} km"
    return f"{distance_m:.0f} m"


def metres_to_km(distance_m: float) -> float:
    """
    Convert metres to kilometres.

    Args:
        distance_m: Distance in metres.

    Returns:
        Distance in kilometres.
    """
    return distance_m / 1000.0


def seconds_to_minutes(seconds: float) -> float:
    """
    Convert seconds to minutes.

    Args:
        seconds: Time in seconds.

    Returns:
        Time in minutes.
    """
    return seconds / 60.0


def elevation_gain(elevations: List[Optional[float]]) -> float:
    """
    Calculate total positive elevation gain from a list of elevation values.

    Ignores None values and negative changes (descents).

    Args:
        elevations: List of elevation values in metres (may contain None).

    Returns:
        Total elevation gain in metres.
    """
    valid = [e for e in elevations if e is not None]
    if len(valid) < 2:
        return 0.0

    gain = sum(
        max(0.0, valid[i + 1] - valid[i])
        for i in range(len(valid) - 1)
    )
    return gain


def elapsed_seconds(
    start: Optional[datetime],
    end: Optional[datetime],
) -> Optional[float]:
    """
    Calculate elapsed seconds between two datetime objects.

    Args:
        start: Start datetime.
        end: End datetime.

    Returns:
        Elapsed time in seconds, or None if either argument is None.
    """
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    if delta < 0:
        logger.warning("End time is before start time; elapsed time is negative.")
    return delta


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """
    Divide two numbers, returning a fallback value if denominator is zero.

    Args:
        numerator: Value to divide.
        denominator: Value to divide by.
        fallback: Return value when denominator is zero (default 0.0).

    Returns:
        Result of division, or fallback.
    """
    if denominator == 0:
        logger.debug("safe_divide: denominator is zero; returning fallback.")
        return fallback
    return numerator / denominator


def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    Clamp a value between a minimum and maximum bound.

    Args:
        value: Input value.
        minimum: Lower bound.
        maximum: Upper bound.

    Returns:
        Clamped value.
    """
    return max(minimum, min(maximum, value))


def deduplicate_timestamps(
    timestamps: List[Optional[datetime]],
) -> List[Optional[datetime]]:
    """
    Remove duplicate consecutive timestamps by setting duplicates to None.

    Args:
        timestamps: List of datetime objects (may contain None).

    Returns:
        List with consecutive duplicates replaced by None.
    """
    cleaned: List[Optional[datetime]] = []
    prev: Optional[datetime] = None

    for ts in timestamps:
        if ts is not None and ts == prev:
            logger.warning(f"Duplicate timestamp detected: {ts}. Setting to None.")
            cleaned.append(None)
        else:
            cleaned.append(ts)
            prev = ts

    return cleaned


# --- Example usage ---
if __name__ == "__main__":
    print(format_pace(330))          # 5:30 /km
    print(format_pace(0))            # --:-- /km
    print(format_duration(5025))     # 1:23:45
    print(format_distance(10500))    # 10.50 km
    print(format_distance(850))      # 850 m
    print(elevation_gain([100.0, 105.0, 103.0, 110.0, None, 108.0]))  # 12.0
    print(safe_divide(10, 0))        # 0.0
    print(clamp(15.0, 0.0, 10.0))    # 10.0

    from datetime import datetime
    t1 = datetime(2024, 1, 1, 8, 0, 0)
    t2 = datetime(2024, 1, 1, 9, 30, 0)
    print(elapsed_seconds(t1, t2))   # 5400.0

    dupe_times = [t1, t1, t2]
    print(deduplicate_timestamps(dupe_times))
