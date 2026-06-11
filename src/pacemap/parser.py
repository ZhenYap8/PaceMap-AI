"""
parser.py

Responsible for:
- Loading GPX files
- Extracting coordinates
- Extracting timestamps
- Extracting elevation data
"""

import gpxpy
import gpxpy.gpx
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackPoint:
    """Represents a single point in a GPX track."""
    latitude: float
    longitude: float
    elevation: Optional[float]
    timestamp: Optional[datetime]


def load_gpx_file(filepath: str) -> gpxpy.gpx.GPX:
    """
    Load and parse a GPX file from disk.

    Args:
        filepath: Absolute or relative path to the .gpx file.

    Returns:
        A parsed GPX object.

    Raises:
        FileNotFoundError: If the file does not exist.
        gpxpy.gpx.GPXException: If the file is malformed.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
        logger.info(f"Successfully loaded GPX file: {filepath}")
        return gpx
    except FileNotFoundError:
        logger.error(f"GPX file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse GPX file '{filepath}': {e}")
        raise


def extract_track_points(gpx: gpxpy.gpx.GPX) -> List[TrackPoint]:
    """
    Extract all track points from a parsed GPX object.

    Skips points with missing latitude or longitude.
    Warns about missing timestamps or elevation values.

    Args:
        gpx: A parsed GPX object.

    Returns:
        A list of TrackPoint dataclass instances.
    """
    points: List[TrackPoint] = []

    def _append_point(point) -> None:
        if point.latitude is None or point.longitude is None:
            logger.warning("Skipping point with missing coordinates.")
            return

        if point.time is None:
            logger.debug(
                f"Point at ({point.latitude}, {point.longitude}) has no timestamp."
            )

        points.append(
            TrackPoint(
                latitude=point.latitude,
                longitude=point.longitude,
                elevation=point.elevation,
                timestamp=point.time,
            )
        )

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                _append_point(point)

    # Some GPX files store planned routes as <rte> instead of <trk>
    for route in gpx.routes:
        for point in route.points:
            _append_point(point)

    logger.info(f"Extracted {len(points)} track points.")
    return points


def extract_coordinates(points: List[TrackPoint]) -> List[tuple[float, float]]:
    """
    Extract (latitude, longitude) tuples from a list of TrackPoints.

    Args:
        points: List of TrackPoint instances.

    Returns:
        List of (latitude, longitude) tuples.
    """
    return [(p.latitude, p.longitude) for p in points]


def extract_timestamps(points: List[TrackPoint]) -> List[Optional[datetime]]:
    """
    Extract timestamps from a list of TrackPoints.

    Args:
        points: List of TrackPoint instances.

    Returns:
        List of datetime objects (or None where missing).
    """
    return [p.timestamp for p in points]


def _extension_value(point, tag_suffixes: tuple[str, ...]) -> Optional[float]:
    """Extract a numeric value from GPX point extensions (Garmin, Strava, etc.)."""
    if not point.extensions:
        return None

    for ext in point.extensions:
        for child in ext:
            tag = child.tag.split("}")[-1].lower() if "}" in child.tag else child.tag.lower()
            if tag in tag_suffixes and child.text:
                try:
                    return float(child.text)
                except ValueError:
                    continue
    return None


def extract_heart_rates(gpx: gpxpy.gpx.GPX) -> List[Optional[float]]:
    """
    Extract heart rate values from GPX track point extensions.

    Supports common Garmin/Strava extension tags (hr, heartrate).
    """
    rates: List[Optional[float]] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rates.append(_extension_value(point, ("hr", "heartrate")))
    return rates


def extract_cadences(gpx: gpxpy.gpx.GPX) -> List[Optional[float]]:
    """
    Extract cadence values from GPX track point extensions.

    Supports common Garmin/Strava extension tags (cad, cadence).
    """
    cadences: List[Optional[float]] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                cadences.append(_extension_value(point, ("cad", "cadence")))
    return cadences


def extract_elevations(points: List[TrackPoint]) -> List[Optional[float]]:
    """
    Extract elevation values from a list of TrackPoints.

    Args:
        points: List of TrackPoint instances.

    Returns:
        List of elevation values in metres (or None where missing).
    """
    return [p.elevation for p in points]


# --- Example usage ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_file.gpx>")
        sys.exit(1)

    gpx_data = load_gpx_file(sys.argv[1])
    track_points = extract_track_points(gpx_data)

    print(f"Total points: {len(track_points)}")
    print(f"First point : {track_points[0] if track_points else 'None'}")
    print(f"Last point  : {track_points[-1] if track_points else 'None'}")
