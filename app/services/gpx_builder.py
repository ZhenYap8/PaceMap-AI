"""
Build GPX files from generated route coordinates.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pacemap.pace_calculator import calculate_segment_distances


def write_route_gpx_with_distance(
    coordinates: list[tuple[float, float]],
    output_path: str,
    route_name: str,
    distance_km: float,
    pace_s_per_km: float,
    elevations: list[float | None] | None = None,
) -> str:
    """Write GPX with timestamps spaced by segment distance at the given pace."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    segment_distances = calculate_segment_distances(coordinates)

    start_time = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    elapsed_s = 0.0
    points_xml = []

    for i, (lat, lon) in enumerate(coordinates):
        elev = (
            elevations[i]
            if elevations and i < len(elevations) and elevations[i] is not None
            else 10.0
        )
        ts = start_time + timedelta(seconds=elapsed_s)
        points_xml.append(
            f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">'
            f"<ele>{elev:.1f}</ele>"
            f"<time>{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}</time>"
            f"</trkpt>"
        )
        if i < len(segment_distances):
            seg_km = max(segment_distances[i] / 1000, 0.001)
            elapsed_s += max(seg_km * pace_s_per_km, 1.0)
        elif i > 0:
            elapsed_s += 1.0

    gpx = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="PaceMap-AI">
  <trk>
    <name>{route_name}</name>
    <trkseg>
{chr(10).join(points_xml)}
    </trkseg>
  </trk>
</gpx>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(gpx)
    return output_path
