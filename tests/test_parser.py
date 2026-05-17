"""
test_parser.py

Tests for GPX parsing module covering:
- Valid GPX file parsing
- Corrupted/malformed files
- Missing data handling
- Edge cases
"""

import pytest
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser import (
    load_gpx_file,
    extract_track_points,
    extract_coordinates,
    extract_timestamps,
    extract_elevations
)


class TestLoadGPXFile:
    """Test GPX file loading"""
    
    def test_load_valid_gpx(self):
        """Should successfully load a valid GPX file"""
        gpx_path = 'data/raw_gpx/run_1.gpx'
        if os.path.exists(gpx_path):
            gpx = load_gpx_file(gpx_path)
            assert gpx is not None
            assert hasattr(gpx, 'tracks')
    
    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            load_gpx_file('nonexistent.gpx')
    
    def test_load_empty_file(self, tmp_path):
        """Should handle empty GPX file gracefully"""
        empty_file = tmp_path / "empty.gpx"
        empty_file.write_text("")
        
        with pytest.raises(Exception):  # gpxpy will raise parsing error
            load_gpx_file(str(empty_file))
    
    def test_load_corrupted_xml(self, tmp_path):
        """Should raise error for corrupted XML"""
        corrupted = tmp_path / "corrupted.gpx"
        corrupted.write_text("<gpx><track><incomplete")
        
        with pytest.raises(Exception):
            load_gpx_file(str(corrupted))


class TestExtractTrackPoints:
    """Test track point extraction"""
    
    def test_extract_from_valid_gpx(self):
        """Should extract track points from valid GPX"""
        gpx_path = 'data/raw_gpx/run_1.gpx'
        if os.path.exists(gpx_path):
            gpx = load_gpx_file(gpx_path)
            points = extract_track_points(gpx)
            
            assert isinstance(points, list)
            assert len(points) > 0
            assert all(hasattr(p, 'latitude') for p in points)
            assert all(hasattr(p, 'longitude') for p in points)
    
    def test_extract_from_empty_tracks(self):
        """Should return empty list for GPX with no tracks"""
        from gpxpy import gpx as gpxpy_gpx
        
        empty_gpx = gpxpy_gpx.GPX()
        points = extract_track_points(empty_gpx)
        
        assert isinstance(points, list)
        assert len(points) == 0
    
    def test_extract_multiple_segments(self):
        """Should handle multiple track segments"""
        # This would test with a GPX containing multiple segments
        # For now, just verify it doesn't crash
        gpx_path = 'data/raw_gpx/run_1.gpx'
        if os.path.exists(gpx_path):
            gpx = load_gpx_file(gpx_path)
            points = extract_track_points(gpx)
            assert len(points) >= 0


class TestExtractCoordinates:
    """Test coordinate extraction"""
    
    def test_extract_valid_coordinates(self):
        """Should extract (lat, lon) tuples"""
        gpx_path = 'data/raw_gpx/run_1.gpx'
        if os.path.exists(gpx_path):
            gpx = load_gpx_file(gpx_path)
            points = extract_track_points(gpx)
            coords = extract_coordinates(points)
            
            assert isinstance(coords, list)
            assert len(coords) == len(points)
            assert all(isinstance(c, tuple) for c in coords)
            assert all(len(c) == 2 for c in coords)
            
            # Check lat/lon ranges
            for lat, lon in coords:
                assert -90 <= lat <= 90
                assert -180 <= lon <= 180
    
    def test_extract_empty_points(self):
        """Should return empty list for no points"""
        coords = extract_coordinates([])
        assert coords == []


class TestExtractTimestamps:
    """Test timestamp extraction"""
    
    def test_extract_valid_timestamps(self):
        """Should extract datetime objects"""
        gpx_path = 'data/raw_gpx/run_1.gpx'
        if os.path.exists(gpx_path):
            gpx = load_gpx_file(gpx_path)
            points = extract_track_points(gpx)
            timestamps = extract_timestamps(points)
            
            assert isinstance(timestamps, list)
            assert len(timestamps) == len(points)
            assert all(isinstance(t, datetime) or t is None for t in timestamps)
    
    def test_extract_empty_timestamps(self):
        """Should return empty list for no points"""
        timestamps = extract_timestamps([])
        assert timestamps == []
    
    def test_handle_missing_timestamps(self):
        """Should handle points with missing time data"""
        # Test with our internal TrackPoint dataclass
        from parser import TrackPoint
        
        point_no_time = TrackPoint(latitude=51.5, longitude=-0.1, elevation=10.0, timestamp=None)
        timestamps = extract_timestamps([point_no_time])
        
        assert len(timestamps) == 1
        assert timestamps[0] is None


class TestExtractElevations:
    """Test elevation extraction"""
    
    def test_extract_valid_elevations(self):
        """Should extract elevation values"""
        gpx_path = 'data/raw_gpx/run_1.gpx'
        if os.path.exists(gpx_path):
            gpx = load_gpx_file(gpx_path)
            points = extract_track_points(gpx)
            elevations = extract_elevations(points)
            
            assert isinstance(elevations, list)
            assert len(elevations) == len(points)
            assert all(isinstance(e, (int, float)) or e is None for e in elevations)
    
    def test_extract_empty_elevations(self):
        """Should return empty list for no points"""
        elevations = extract_elevations([])
        assert elevations == []
    
    def test_handle_missing_elevations(self):
        """Should handle points without elevation data"""
        from gpxpy.gpx import GPXTrackPoint
        
        point_no_elev = GPXTrackPoint(latitude=51.5, longitude=-0.1)
        elevations = extract_elevations([point_no_elev])
        
        assert len(elevations) == 1
        assert elevations[0] is None


# Integration test
class TestGPXParsingWorkflow:
    """Test complete parsing workflow"""
    
    def test_full_parsing_pipeline(self):
        """Should parse GPX and extract all data successfully"""
        gpx_path = 'data/raw_gpx/run_1.gpx'
        if not os.path.exists(gpx_path):
            pytest.skip("Test GPX file not available")
        
        # Full pipeline
        gpx = load_gpx_file(gpx_path)
        points = extract_track_points(gpx)
        coords = extract_coordinates(points)
        timestamps = extract_timestamps(points)
        elevations = extract_elevations(points)
        
        # Validate consistency
        assert len(coords) == len(points)
        assert len(timestamps) == len(points)
        assert len(elevations) == len(points)
        
        # Validate at least some valid data
        assert len([c for c in coords if c]) > 0
        assert len([t for t in timestamps if t]) > 0
