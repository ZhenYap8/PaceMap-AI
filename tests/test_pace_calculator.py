"""
test_pace_calculator.py

Tests for pace_calculator module covering:
- Haversine distance calculations
- Segment distance calculations
- Cumulative distance tracking
- Speed and pace calculations
- Moving average smoothing
- GPS coordinate smoothing
- Edge cases and error handling
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pace_calculator import (
    haversine_distance,
    calculate_segment_distances,
    calculate_cumulative_distance,
    calculate_speed,
    calculate_pace,
    calculate_segment_paces,
    moving_average,
    smooth_gps_coordinates,
    EARTH_RADIUS_M
)


class TestHaversineDistance:
    """Test Haversine distance calculations"""
    
    def test_same_point(self):
        """Distance between same point should be 0"""
        dist = haversine_distance(51.5074, -0.1278, 51.5074, -0.1278)
        assert dist == 0.0
    
    def test_known_distance(self):
        """Test with known distance (London to Paris ~344 km)"""
        # London: 51.5074° N, 0.1278° W
        # Paris: 48.8566° N, 2.3522° E
        dist = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        
        # Should be approximately 344 km (allow 5% tolerance)
        assert 327_000 < dist < 361_000
    
    def test_short_distance(self):
        """Test short distance (should be a few hundred meters)"""
        # Two nearby points in London
        dist = haversine_distance(51.5074, -0.1278, 51.5080, -0.1265)
        
        # Should be roughly 100-150 meters
        assert 50 < dist < 200
    
    def test_equator_distance(self):
        """Test distance along equator"""
        # 1 degree longitude at equator ≈ 111 km
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        
        # Should be approximately 111 km
        assert 110_000 < dist < 112_000
    
    def test_negative_coordinates(self):
        """Should handle negative lat/lon (southern/western hemispheres)"""
        dist = haversine_distance(-33.9249, 18.4241, -34.9249, 19.4241)
        assert dist > 0
    
    def test_crosses_antimeridian(self):
        """Should handle coordinates crossing the antimeridian"""
        dist = haversine_distance(0, 179, 0, -179)
        # Should be ~222 km (2 degrees at equator)
        assert 220_000 < dist < 224_000


class TestCalculateSegmentDistances:
    """Test segment distance calculations"""
    
    def test_two_points(self):
        """Should calculate distance for two points"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        distances = calculate_segment_distances(coords)
        
        assert len(distances) == 1
        assert distances[0] > 0
    
    def test_multiple_points(self):
        """Should calculate distances for multiple segments"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5090, -0.1250)
        ]
        distances = calculate_segment_distances(coords)
        
        assert len(distances) == 2
        assert all(d > 0 for d in distances)
    
    def test_empty_list(self):
        """Should return empty list for no coordinates"""
        assert calculate_segment_distances([]) == []
    
    def test_single_point(self):
        """Should return empty list for single point"""
        coords = [(51.5074, -0.1278)]
        assert calculate_segment_distances(coords) == []
    
    def test_stationary_points(self):
        """Should return 0 for identical consecutive points"""
        coords = [(51.5074, -0.1278), (51.5074, -0.1278)]
        distances = calculate_segment_distances(coords)
        
        assert len(distances) == 1
        assert distances[0] == 0.0


class TestCalculateCumulativeDistance:
    """Test cumulative distance calculations"""
    
    def test_normal_segments(self):
        """Should calculate cumulative distances correctly"""
        segments = [100.0, 200.0, 150.0]
        cumulative = calculate_cumulative_distance(segments)
        
        assert cumulative == [0.0, 100.0, 300.0, 450.0]
    
    def test_empty_segments(self):
        """Should return [0.0] for empty input"""
        cumulative = calculate_cumulative_distance([])
        assert cumulative == [0.0]
    
    def test_single_segment(self):
        """Should handle single segment"""
        cumulative = calculate_cumulative_distance([500.0])
        assert cumulative == [0.0, 500.0]
    
    def test_zero_distances(self):
        """Should handle zero distances"""
        segments = [0.0, 100.0, 0.0, 50.0]
        cumulative = calculate_cumulative_distance(segments)
        assert cumulative == [0.0, 0.0, 100.0, 100.0, 150.0]


class TestCalculateSpeed:
    """Test speed calculations"""
    
    def test_normal_speed(self):
        """Should calculate speed correctly"""
        # 1000m in 200s = 5 m/s
        speed = calculate_speed(1000.0, 200.0)
        assert speed == 5.0
    
    def test_zero_time(self):
        """Should return 0 for zero time"""
        speed = calculate_speed(1000.0, 0.0)
        assert speed == 0.0
    
    def test_negative_time(self):
        """Should return 0 for negative time"""
        speed = calculate_speed(1000.0, -10.0)
        assert speed == 0.0
    
    def test_zero_distance(self):
        """Should return 0 for zero distance"""
        speed = calculate_speed(0.0, 100.0)
        assert speed == 0.0
    
    def test_fractional_values(self):
        """Should handle fractional distances and times"""
        speed = calculate_speed(500.5, 100.1)
        assert speed == pytest.approx(5.0, rel=0.01)


class TestCalculatePace:
    """Test pace calculations"""
    
    def test_normal_pace(self):
        """Should calculate pace correctly"""
        # 1000m in 300s = 300 s/km = 5:00/km
        pace = calculate_pace(1000.0, 300.0)
        assert pace == 300.0
    
    def test_5_min_per_km(self):
        """Should calculate 5 min/km pace"""
        # 5000m in 1500s = 5:00/km
        pace = calculate_pace(5000.0, 1500.0)
        assert pace == 300.0
    
    def test_zero_distance(self):
        """Should return 0 for zero distance"""
        pace = calculate_pace(0.0, 300.0)
        assert pace == 0.0
    
    def test_negative_distance(self):
        """Should return 0 for negative distance"""
        pace = calculate_pace(-1000.0, 300.0)
        assert pace == 0.0
    
    def test_very_fast_pace(self):
        """Should handle very fast paces"""
        # 1000m in 180s = 3:00/km (elite pace)
        pace = calculate_pace(1000.0, 180.0)
        assert pace == 180.0
    
    def test_very_slow_pace(self):
        """Should handle very slow paces"""
        # 1000m in 600s = 10:00/km
        pace = calculate_pace(1000.0, 600.0)
        assert pace == 600.0


class TestCalculateSegmentPaces:
    """Test segment pace calculations"""
    
    def test_normal_segments(self):
        """Should calculate paces for all segments"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5090, -0.1250)
        ]
        base = datetime(2024, 1, 1, 8, 0, 0)
        timestamps = [
            base,
            base + timedelta(seconds=30),
            base + timedelta(seconds=60)
        ]
        
        paces = calculate_segment_paces(coords, timestamps)
        
        assert len(paces) == 2
        assert all(p >= 0 for p in paces)
    
    def test_missing_timestamp(self):
        """Should set pace to 0 for missing timestamps"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        timestamps = [datetime.now(), None]
        
        paces = calculate_segment_paces(coords, timestamps)
        
        assert len(paces) == 1
        assert paces[0] == 0.0
    
    def test_duplicate_timestamp(self):
        """Should set pace to 0 for duplicate timestamps"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        t = datetime.now()
        timestamps = [t, t]
        
        paces = calculate_segment_paces(coords, timestamps)
        
        assert len(paces) == 1
        assert paces[0] == 0.0
    
    def test_backwards_time(self):
        """Should set pace to 0 for backwards timestamps"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        base = datetime(2024, 1, 1, 8, 0, 0)
        timestamps = [base, base - timedelta(seconds=10)]
        
        paces = calculate_segment_paces(coords, timestamps)
        
        assert len(paces) == 1
        assert paces[0] == 0.0
    
    def test_mismatched_lengths(self):
        """Should raise error for mismatched coordinate/timestamp lengths"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        timestamps = [datetime.now()]
        
        with pytest.raises(ValueError, match="must have the same length"):
            calculate_segment_paces(coords, timestamps)
    
    def test_empty_lists(self):
        """Should return empty list for no coordinates"""
        paces = calculate_segment_paces([], [])
        assert paces == []


class TestMovingAverage:
    """Test moving average smoothing"""
    
    def test_normal_smoothing(self):
        """Should smooth values correctly"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        smoothed = moving_average(values, window=3)
        
        assert len(smoothed) == len(values)
        # Middle value should be average of neighbors
        assert smoothed[2] == 3.0
    
    def test_window_1(self):
        """Window of 1 should return original values"""
        values = [1.0, 2.0, 3.0]
        smoothed = moving_average(values, window=1)
        assert smoothed == values
    
    def test_large_window(self):
        """Large window should heavily smooth"""
        values = [1.0, 10.0, 1.0, 10.0, 1.0]
        smoothed = moving_average(values, window=5)
        
        # All values should be closer to the mean
        mean_val = sum(values) / len(values)
        assert all(abs(v - mean_val) < 5 for v in smoothed)
    
    def test_edge_values(self):
        """Edge values should use smaller windows"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        smoothed = moving_average(values, window=5)
        
        # First and last values should be different from middle
        assert smoothed[0] != smoothed[2]
        assert smoothed[4] != smoothed[2]
    
    def test_invalid_window(self):
        """Should raise error for window < 1"""
        with pytest.raises(ValueError, match="window must be >= 1"):
            moving_average([1.0, 2.0, 3.0], window=0)
    
    def test_empty_list(self):
        """Should handle empty list"""
        smoothed = moving_average([], window=3)
        assert smoothed == []
    
    def test_single_value(self):
        """Should handle single value"""
        smoothed = moving_average([5.0], window=3)
        assert smoothed == [5.0]


class TestSmoothGPSCoordinates:
    """Test GPS coordinate smoothing"""
    
    def test_normal_smoothing(self):
        """Should smooth GPS coordinates"""
        coords = [
            (51.5074, -0.1278),
            (51.5075, -0.1277),
            (51.5076, -0.1276),
            (51.5077, -0.1275)
        ]
        smoothed = smooth_gps_coordinates(coords, window=3)
        
        assert len(smoothed) == len(coords)
        # All should still be valid lat/lon
        for lat, lon in smoothed:
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180
    
    def test_noisy_gps(self):
        """Should reduce noise in GPS data"""
        # Simulated noisy GPS with one outlier
        coords = [
            (51.5074, -0.1278),
            (51.5075, -0.1277),
            (51.5099, -0.1250),  # Outlier
            (51.5076, -0.1276),
            (51.5077, -0.1275)
        ]
        smoothed = smooth_gps_coordinates(coords, window=3)
        
        # Outlier should be reduced
        assert abs(smoothed[2][0] - coords[2][0]) > 0.001
    
    def test_empty_coordinates(self):
        """Should return empty list for no coordinates"""
        smoothed = smooth_gps_coordinates([], window=3)
        assert smoothed == []
    
    def test_single_coordinate(self):
        """Should return same coordinate for single point"""
        coords = [(51.5074, -0.1278)]
        smoothed = smooth_gps_coordinates(coords, window=3)
        assert smoothed == coords
    
    def test_preserves_general_path(self):
        """Should preserve general path shape"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1270),
            (51.5086, -0.1262)
        ]
        smoothed = smooth_gps_coordinates(coords, window=3)
        
        # First lat should still be smallest, last largest
        assert smoothed[0][0] < smoothed[2][0]
        # First lon should be smallest, last largest
        assert smoothed[0][1] < smoothed[2][1]


class TestPaceCalculatorIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_run_calculation(self):
        """Should calculate all metrics for a complete run"""
        # Simulate a 1km run with 5 points
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252),
            (51.5092, -0.1239),
            (51.5098, -0.1226)
        ]
        
        base = datetime(2024, 1, 1, 8, 0, 0)
        timestamps = [base + timedelta(seconds=i * 60) for i in range(5)]
        
        # Calculate all metrics
        distances = calculate_segment_distances(coords)
        cumulative = calculate_cumulative_distance(distances)
        paces = calculate_segment_paces(coords, timestamps)
        smoothed_coords = smooth_gps_coordinates(coords, window=3)
        
        # Validate results
        assert len(distances) == 4
        assert len(cumulative) == 5
        assert len(paces) == 4
        assert len(smoothed_coords) == 5
        
        assert cumulative[0] == 0.0
        assert cumulative[-1] > 0
        assert all(p > 0 for p in paces)
    
    def test_stationary_run(self):
        """Should handle stationary GPS points"""
        # Same location repeated
        coords = [(51.5074, -0.1278)] * 5
        base = datetime.now()
        timestamps = [base + timedelta(seconds=i * 10) for i in range(5)]
        
        distances = calculate_segment_distances(coords)
        paces = calculate_segment_paces(coords, timestamps)
        
        # All distances should be 0
        assert all(d == 0.0 for d in distances)
        # All paces should be 0
        assert all(p == 0.0 for p in paces)
    
    def test_realistic_5k_run(self):
        """Should produce realistic metrics for a 5K run"""
        # Simulate 5K run with 50 points (100m apart)
        coords = [(51.5074 + i * 0.001, -0.1278 + i * 0.0005) for i in range(51)]
        
        # 25 minutes total (5:00/km pace)
        base = datetime(2024, 1, 1, 8, 0, 0)
        timestamps = [base + timedelta(seconds=i * 30) for i in range(51)]
        
        distances = calculate_segment_distances(coords)
        cumulative = calculate_cumulative_distance(distances)
        paces = calculate_segment_paces(coords, timestamps)
        
        # Total distance should be roughly 5-6 km
        total_distance = cumulative[-1]
        assert 4_000 < total_distance < 7_000
        
        # Average pace should be reasonable (3:00 - 8:00 /km)
        valid_paces = [p for p in paces if p > 0]
        if valid_paces:
            avg_pace = sum(valid_paces) / len(valid_paces)
            assert 180 < avg_pace < 480


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
