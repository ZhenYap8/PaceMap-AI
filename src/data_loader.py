"""
data_loader.py

Loads all GPX files and extracts features for ML training.
"""

import logging
import os
from typing import Dict, Optional
import pandas as pd

from parser import load_gpx_file, extract_track_points, extract_coordinates, extract_timestamps, extract_elevations
from pace_calculator import smooth_gps_coordinates, calculate_segment_distances, calculate_cumulative_distance, calculate_segment_paces
from utils import elevation_gain, elapsed_seconds, metres_to_km, deduplicate_timestamps

logger = logging.getLogger(__name__)


def extract_run_features(gpx_filepath: str, smoothing_window: int = 3) -> Optional[Dict]:
    """
    Extract all features from a single GPX file.
    
    Args:
        gpx_filepath: Path to the GPX file.
        smoothing_window: Window size for GPS smoothing.
    
    Returns:
        Dictionary of features for ML training, or None if extraction fails.
    """
    try:
        # Parse GPX
        gpx = load_gpx_file(gpx_filepath)
        track_points = extract_track_points(gpx)
        
        if len(track_points) < 2:
            logger.warning(f"Insufficient track points in {gpx_filepath}")
            return None
        
        coordinates = extract_coordinates(track_points)
        timestamps = extract_timestamps(track_points)
        elevations = extract_elevations(track_points)
        
        # Calculate features
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
        
        if total_elapsed is None or total_elapsed <= 0:
            logger.warning(f"Invalid elapsed time for {gpx_filepath}")
            return None
        
        return {
            'distance_km': metres_to_km(total_distance_m),
            'elevation_gain_m': elev_gain,
            'avg_pace_s_per_km': avg_pace,
            'avg_heart_rate_bpm': 155.0,  # Placeholder - update if available in GPX
            'temperature_c': 20.0,  # Placeholder - could add weather API
            'time_of_day_hour': timestamps[0].hour if timestamps[0] else 8,
            'fatigue_score': 5.0,  # Placeholder - could add manual logging
            'finish_time_s': total_elapsed,
        }
    
    except Exception as e:
        logger.error(f"Failed to process {gpx_filepath}: {e}")
        return None


def load_all_runs(gpx_dir: str, smoothing_window: int = 3) -> pd.DataFrame:
    """
    Load all GPX files from a directory and extract features.
    
    Args:
        gpx_dir: Directory containing GPX files (run_1.gpx, run_2.gpx, etc.).
        smoothing_window: Window size for GPS smoothing.
    
    Returns:
        DataFrame with features for all runs.
    """
    all_runs = []
    
    # Get all GPX files and sort them
    gpx_files = sorted([f for f in os.listdir(gpx_dir) if f.endswith('.gpx')])
    logger.info(f"Found {len(gpx_files)} GPX files in {gpx_dir}")
    
    for gpx_file in gpx_files:
        gpx_path = os.path.join(gpx_dir, gpx_file)
        logger.info(f"Processing {gpx_file}...")
        
        features = extract_run_features(gpx_path, smoothing_window)
        if features:
            features['run_id'] = gpx_file  # Keep track of which run
            all_runs.append(features)
        else:
            logger.warning(f"Skipping {gpx_file} due to extraction errors")
    
    df = pd.DataFrame(all_runs)
    logger.info(f"Successfully loaded {len(df)} runs")
    
    return df
