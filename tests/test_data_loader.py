"""
test_data_loader.py

Tests for data_loader module covering:
- Single GPX file feature extraction
- Batch GPX file loading
- Error handling for corrupted files
- Edge cases (empty files, missing data)
"""

import pytest
import sys
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import extract_run_features, load_all_runs


# Sample GPX content for testing
VALID_GPX_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="TestApp">
  <trk>
    <name>Test Run</name>
    <trkseg>
      <trkpt lat="51.5074" lon="-0.1278">
        <ele>10.0</ele>
        <time>2024-01-01T08:00:00Z</time>
      </trkpt>
      <trkpt lat="51.5080" lon="-0.1265">
        <ele>12.0</ele>
        <time>2024-01-01T08:01:00Z</time>
      </trkpt>
      <trkpt lat="51.5086" lon="-0.1252">
        <ele>15.0</ele>
        <time>2024-01-01T08:02:00Z</time>
      </trkpt>
      <trkpt lat="51.5092" lon="-0.1239">
        <ele>14.0</ele>
        <time>2024-01-01T08:03:00Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>
"""

EMPTY_GPX_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="TestApp">
  <trk>
    <name>Empty Run</name>
  </trk>
</gpx>
"""

SINGLE_POINT_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="TestApp">
  <trk>
    <trkseg>
      <trkpt lat="51.5074" lon="-0.1278">
        <ele>10.0</ele>
        <time>2024-01-01T08:00:00Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>
"""


class TestExtractRunFeatures:
    """Test single GPX file feature extraction"""
    
    def test_extract_valid_gpx(self, tmp_path):
        """Should extract features from valid GPX file"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        assert features is not None
        assert 'distance_km' in features
        assert 'elevation_gain_m' in features
        assert 'avg_pace_s_per_km' in features
        assert 'finish_time_s' in features
        assert features['distance_km'] > 0
        assert features['finish_time_s'] > 0
    
    def test_extract_feature_types(self, tmp_path):
        """Should return correct data types"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        assert isinstance(features['distance_km'], float)
        assert isinstance(features['elevation_gain_m'], float)
        assert isinstance(features['avg_pace_s_per_km'], (int, float))
        assert isinstance(features['finish_time_s'], (int, float))
        assert isinstance(features['time_of_day_hour'], int)
    
    def test_extract_calculates_distance(self, tmp_path):
        """Should calculate reasonable distance"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        # 4 points in London, roughly 300m apart = ~0.9km total
        assert 0.2 < features['distance_km'] < 2.0
    
    def test_extract_calculates_elevation_gain(self, tmp_path):
        """Should calculate elevation gain correctly"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        # Elevations: 10 -> 12 -> 15 -> 14
        # Gains: +2, +3, -1 (ignored) = 5m total
        assert features['elevation_gain_m'] == 5.0
    
    def test_extract_calculates_duration(self, tmp_path):
        """Should calculate correct finish time"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        # 3 minutes between first and last point
        assert features['finish_time_s'] == 180.0
    
    def test_extract_with_custom_smoothing(self, tmp_path):
        """Should respect custom smoothing window"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file), smoothing_window=5)
        
        assert features is not None
        assert features['distance_km'] > 0
    
    def test_extract_insufficient_points(self, tmp_path):
        """Should return None for insufficient track points"""
        gpx_file = tmp_path / "single_point.gpx"
        gpx_file.write_text(SINGLE_POINT_GPX)
        
        features = extract_run_features(str(gpx_file))
        
        assert features is None
    
    def test_extract_empty_gpx(self, tmp_path):
        """Should return None for empty GPX"""
        gpx_file = tmp_path / "empty.gpx"
        gpx_file.write_text(EMPTY_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        assert features is None
    
    def test_extract_nonexistent_file(self):
        """Should return None for nonexistent file"""
        features = extract_run_features("/nonexistent/path/run.gpx")
        
        assert features is None
    
    def test_extract_corrupted_gpx(self, tmp_path):
        """Should return None for corrupted GPX"""
        gpx_file = tmp_path / "corrupted.gpx"
        gpx_file.write_text("This is not valid XML")
        
        features = extract_run_features(str(gpx_file))
        
        assert features is None
    
    def test_extract_includes_placeholder_fields(self, tmp_path):
        """Should include placeholder fields"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        assert 'avg_heart_rate_bpm' in features
        assert 'temperature_c' in features
        assert 'fatigue_score' in features
        assert features['avg_heart_rate_bpm'] == 155.0
        assert features['temperature_c'] == 20.0
        assert features['fatigue_score'] == 5.0
    
    def test_extract_time_of_day(self, tmp_path):
        """Should extract correct time of day"""
        gpx_file = tmp_path / "test_run.gpx"
        gpx_file.write_text(VALID_GPX_CONTENT)
        
        features = extract_run_features(str(gpx_file))
        
        # First timestamp is 08:00
        assert features['time_of_day_hour'] == 8


class TestLoadAllRuns:
    """Test batch GPX file loading"""
    
    def test_load_multiple_runs(self, tmp_path):
        """Should load multiple GPX files"""
        # Create 3 valid GPX files
        for i in range(1, 4):
            gpx_file = tmp_path / f"run_{i}.gpx"
            gpx_file.write_text(VALID_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'distance_km' in df.columns
        assert 'run_id' in df.columns
    
    def test_load_empty_directory(self, tmp_path):
        """Should return empty DataFrame for empty directory"""
        df = load_all_runs(str(tmp_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_load_skips_invalid_files(self, tmp_path):
        """Should skip invalid GPX files"""
        # Create 2 valid and 1 invalid
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_2.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_3.gpx").write_text(EMPTY_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path))
        
        assert len(df) == 2  # Only valid files loaded
    
    def test_load_includes_run_ids(self, tmp_path):
        """Should include run IDs in DataFrame"""
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_2.gpx").write_text(VALID_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path))
        
        assert 'run_id' in df.columns
        assert 'run_1.gpx' in df['run_id'].values
        assert 'run_2.gpx' in df['run_id'].values
    
    def test_load_with_custom_smoothing(self, tmp_path):
        """Should respect custom smoothing window"""
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path), smoothing_window=5)
        
        assert len(df) == 1
        assert df.iloc[0]['distance_km'] > 0
    
    def test_load_sorts_files(self, tmp_path):
        """Should load files in sorted order"""
        # Create files out of order
        (tmp_path / "run_10.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_2.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path))
        
        # Should be sorted: run_1, run_10, run_2 (alphabetically)
        assert df.iloc[0]['run_id'] == 'run_1.gpx'
    
    def test_load_ignores_non_gpx_files(self, tmp_path):
        """Should ignore non-GPX files"""
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "readme.txt").write_text("This is a text file")
        (tmp_path / "data.csv").write_text("col1,col2\n1,2")
        
        df = load_all_runs(str(tmp_path))
        
        assert len(df) == 1
        assert df.iloc[0]['run_id'] == 'run_1.gpx'
    
    def test_load_returns_dataframe_with_all_columns(self, tmp_path):
        """Should return DataFrame with all expected columns"""
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path))
        
        expected_columns = [
            'distance_km',
            'elevation_gain_m',
            'avg_pace_s_per_km',
            'avg_heart_rate_bpm',
            'temperature_c',
            'time_of_day_hour',
            'fatigue_score',
            'finish_time_s',
            'run_id'
        ]
        
        for col in expected_columns:
            assert col in df.columns


class TestDataLoaderIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_data_loading_pipeline(self, tmp_path):
        """Should load and process multiple runs correctly"""
        # Create a realistic set of GPX files
        for i in range(1, 6):
            gpx_file = tmp_path / f"run_{i}.gpx"
            gpx_file.write_text(VALID_GPX_CONTENT)
        
        # Load all runs
        df = load_all_runs(str(tmp_path))
        
        # Validate DataFrame
        assert len(df) == 5
        assert not df.empty
        assert df['distance_km'].min() > 0
        assert df['finish_time_s'].min() > 0
        assert df['elevation_gain_m'].min() >= 0
        
        # All runs should have valid data
        assert not df['distance_km'].isna().any()
        assert not df['finish_time_s'].isna().any()
    
    def test_mixed_valid_invalid_runs(self, tmp_path):
        """Should handle mix of valid and invalid files"""
        # Create mix of valid and invalid files
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_2.gpx").write_text(EMPTY_GPX_CONTENT)
        (tmp_path / "run_3.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_4.gpx").write_text(SINGLE_POINT_GPX)
        (tmp_path / "run_5.gpx").write_text(VALID_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path))
        
        # Should load only the 3 valid files
        assert len(df) == 3
        assert all(df['distance_km'] > 0)
    
    def test_data_ready_for_ml(self, tmp_path):
        """Should produce data ready for ML model"""
        (tmp_path / "run_1.gpx").write_text(VALID_GPX_CONTENT)
        (tmp_path / "run_2.gpx").write_text(VALID_GPX_CONTENT)
        
        df = load_all_runs(str(tmp_path))
        
        # Remove run_id for ML
        X = df.drop(columns=['run_id', 'finish_time_s'])
        y = df['finish_time_s']
        
        assert len(X) == 2
        assert len(y) == 2
        assert not X.isna().any().any()
        assert not y.isna().any()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
