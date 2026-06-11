"""
test_utils.py

Tests for utility functions covering:
- Formatting functions (pace, duration, distance)
- Elevation gain calculation
- Time calculations
- Safe operations (divide, clamp)
- Timestamp deduplication
"""

from datetime import datetime, timedelta

import pytest

from pacemap.utils import (
    format_pace,
    format_duration,
    format_distance,
    metres_to_km,
    seconds_to_minutes,
    elevation_gain,
    elapsed_seconds,
    safe_divide,
    clamp,
    deduplicate_timestamps
)


class TestFormatPace:
    """Test pace formatting"""
    
    def test_format_normal_pace(self):
        """Should format normal pace correctly"""
        assert format_pace(330) == "5:30 /km"
        assert format_pace(300) == "5:00 /km"
        assert format_pace(270) == "4:30 /km"
    
    def test_format_fast_pace(self):
        """Should format fast pace (< 4 min/km)"""
        assert format_pace(240) == "4:00 /km"
        assert format_pace(180) == "3:00 /km"
    
    def test_format_slow_pace(self):
        """Should format slow pace (> 7 min/km)"""
        assert format_pace(420) == "7:00 /km"
        assert format_pace(600) == "10:00 /km"
    
    def test_format_zero_pace(self):
        """Should handle zero pace"""
        assert format_pace(0) == "--:-- /km"
    
    def test_format_negative_pace(self):
        """Should handle negative pace"""
        assert format_pace(-100) == "--:-- /km"
    
    def test_format_fractional_pace(self):
        """Should round fractional seconds"""
        assert format_pace(330.7) == "5:30 /km"
        assert format_pace(329.2) == "5:29 /km"


class TestFormatDuration:
    """Test duration formatting"""
    
    def test_format_seconds_only(self):
        """Should format durations under 1 minute"""
        assert format_duration(45) == "0:00:45"
        assert format_duration(30) == "0:00:30"
    
    def test_format_minutes(self):
        """Should format durations with minutes"""
        assert format_duration(90) == "0:01:30"
        assert format_duration(600) == "0:10:00"
    
    def test_format_hours(self):
        """Should format durations with hours"""
        assert format_duration(3661) == "1:01:01"
        assert format_duration(5025) == "1:23:45"
        assert format_duration(7200) == "2:00:00"
    
    def test_format_zero(self):
        """Should handle zero duration"""
        assert format_duration(0) == "0:00:00"
    
    def test_format_negative(self):
        """Should treat negative as zero"""
        assert format_duration(-100) == "0:00:00"
    
    def test_format_float(self):
        """Should handle float input"""
        assert format_duration(90.8) == "0:01:30"


class TestFormatDistance:
    """Test distance formatting"""
    
    def test_format_metres(self):
        """Should display metres for distances < 1000m"""
        assert format_distance(500) == "500 m"
        assert format_distance(850) == "850 m"
        assert format_distance(999) == "999 m"
    
    def test_format_kilometres(self):
        """Should display km for distances >= 1000m"""
        assert format_distance(1000) == "1.00 km"
        assert format_distance(5250) == "5.25 km"
        assert format_distance(42195) == "42.20 km"  # Marathon
    
    def test_format_zero(self):
        """Should handle zero distance"""
        assert format_distance(0) == "0 m"
    
    def test_format_fractional(self):
        """Should round to 2 decimal places for km"""
        assert format_distance(10567) == "10.57 km"


class TestConversionFunctions:
    """Test unit conversion functions"""
    
    def test_metres_to_km(self):
        """Should convert metres to kilometres"""
        assert metres_to_km(1000) == 1.0
        assert metres_to_km(5250) == 5.25
        assert metres_to_km(0) == 0.0
    
    def test_seconds_to_minutes(self):
        """Should convert seconds to minutes"""
        assert seconds_to_minutes(60) == 1.0
        assert seconds_to_minutes(90) == 1.5
        assert seconds_to_minutes(0) == 0.0


class TestElevationGain:
    """Test elevation gain calculation"""
    
    def test_positive_gain_only(self):
        """Should sum only positive elevation changes"""
        elevations = [100.0, 105.0, 110.0, 115.0]
        assert elevation_gain(elevations) == 15.0
    
    def test_with_descents(self):
        """Should ignore descents (negative changes)"""
        elevations = [100.0, 105.0, 103.0, 110.0]
        # +5, -2 (ignored), +7 = 12
        assert elevation_gain(elevations) == 12.0
    
    def test_with_none_values(self):
        """Should skip None values"""
        elevations = [100.0, 105.0, None, 110.0, 115.0]
        # Ignores None, calculates on [100, 105, 110, 115]
        assert elevation_gain(elevations) == 15.0
    
    def test_all_descents(self):
        """Should return 0 for all descents"""
        elevations = [115.0, 110.0, 105.0, 100.0]
        assert elevation_gain(elevations) == 0.0
    
    def test_single_value(self):
        """Should return 0 for single elevation point"""
        assert elevation_gain([100.0]) == 0.0
    
    def test_empty_list(self):
        """Should return 0 for empty list"""
        assert elevation_gain([]) == 0.0
    
    def test_all_none(self):
        """Should return 0 for all None values"""
        assert elevation_gain([None, None, None]) == 0.0


class TestElapsedSeconds:
    """Test elapsed time calculation"""
    
    def test_normal_elapsed(self):
        """Should calculate elapsed seconds correctly"""
        start = datetime(2024, 1, 1, 8, 0, 0)
        end = datetime(2024, 1, 1, 9, 30, 0)
        assert elapsed_seconds(start, end) == 5400.0  # 1.5 hours
    
    def test_same_time(self):
        """Should return 0 for same start/end"""
        t = datetime(2024, 1, 1, 8, 0, 0)
        assert elapsed_seconds(t, t) == 0.0
    
    def test_negative_elapsed(self):
        """Should handle end before start (returns negative)"""
        start = datetime(2024, 1, 1, 9, 0, 0)
        end = datetime(2024, 1, 1, 8, 0, 0)
        assert elapsed_seconds(start, end) == -3600.0
    
    def test_none_start(self):
        """Should return None if start is None"""
        end = datetime(2024, 1, 1, 9, 0, 0)
        assert elapsed_seconds(None, end) is None
    
    def test_none_end(self):
        """Should return None if end is None"""
        start = datetime(2024, 1, 1, 8, 0, 0)
        assert elapsed_seconds(start, None) is None
    
    def test_both_none(self):
        """Should return None if both are None"""
        assert elapsed_seconds(None, None) is None
    
    def test_fractional_seconds(self):
        """Should handle microseconds"""
        start = datetime(2024, 1, 1, 8, 0, 0, 0)
        end = datetime(2024, 1, 1, 8, 0, 0, 500000)  # +0.5s
        assert elapsed_seconds(start, end) == 0.5


class TestSafeDivide:
    """Test safe division"""
    
    def test_normal_division(self):
        """Should divide normally for non-zero denominator"""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(100, 4) == 25.0
        assert safe_divide(7, 2) == 3.5
    
    def test_divide_by_zero_default(self):
        """Should return default fallback (0.0) for zero denominator"""
        assert safe_divide(10, 0) == 0.0
    
    def test_divide_by_zero_custom_fallback(self):
        """Should return custom fallback for zero denominator"""
        assert safe_divide(10, 0, fallback=999.0) == 999.0
        assert safe_divide(10, 0, fallback=-1.0) == -1.0
    
    def test_zero_numerator(self):
        """Should handle zero numerator"""
        assert safe_divide(0, 5) == 0.0
    
    def test_negative_values(self):
        """Should handle negative numbers"""
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0


class TestClamp:
    """Test value clamping"""
    
    def test_within_bounds(self):
        """Should return value if within bounds"""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(7.5, 5.0, 10.0) == 7.5
    
    def test_below_minimum(self):
        """Should return minimum if value too low"""
        assert clamp(-5.0, 0.0, 10.0) == 0.0
        assert clamp(2.0, 5.0, 10.0) == 5.0
    
    def test_above_maximum(self):
        """Should return maximum if value too high"""
        assert clamp(15.0, 0.0, 10.0) == 10.0
        assert clamp(100.0, 0.0, 10.0) == 10.0
    
    def test_at_boundaries(self):
        """Should handle values at exact boundaries"""
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0
    
    def test_negative_range(self):
        """Should work with negative ranges"""
        assert clamp(-5.0, -10.0, 0.0) == -5.0
        assert clamp(-15.0, -10.0, 0.0) == -10.0


class TestDeduplicateTimestamps:
    """Test timestamp deduplication"""
    
    def test_no_duplicates(self):
        """Should not modify list without duplicates"""
        t1 = datetime(2024, 1, 1, 8, 0, 0)
        t2 = datetime(2024, 1, 1, 8, 0, 1)
        t3 = datetime(2024, 1, 1, 8, 0, 2)
        
        timestamps = [t1, t2, t3]
        result = deduplicate_timestamps(timestamps)
        
        assert result == [t1, t2, t3]
    
    def test_consecutive_duplicates(self):
        """Should replace consecutive duplicates with None"""
        t1 = datetime(2024, 1, 1, 8, 0, 0)
        t2 = datetime(2024, 1, 1, 8, 0, 1)
        
        timestamps = [t1, t1, t2, t2, t2]
        result = deduplicate_timestamps(timestamps)
        
        assert result == [t1, None, t2, None, None]
    
    def test_with_none_values(self):
        """Should preserve existing None values"""
        t1 = datetime(2024, 1, 1, 8, 0, 0)
        t2 = datetime(2024, 1, 1, 8, 0, 1)
        
        timestamps = [t1, None, t2, t2]
        result = deduplicate_timestamps(timestamps)
        
        assert result == [t1, None, t2, None]
    
    def test_empty_list(self):
        """Should handle empty list"""
        assert deduplicate_timestamps([]) == []
    
    def test_all_none(self):
        """Should handle list of all None"""
        timestamps = [None, None, None]
        result = deduplicate_timestamps(timestamps)
        assert result == [None, None, None]
    
    def test_all_same(self):
        """Should handle all same timestamps"""
        t = datetime(2024, 1, 1, 8, 0, 0)
        timestamps = [t, t, t, t]
        result = deduplicate_timestamps(timestamps)
        
        assert result == [t, None, None, None]


# Integration test
class TestUtilsIntegration:
    """Test utilities working together"""
    
    def test_complete_run_formatting(self):
        """Should format all run statistics correctly"""
        # Simulate a 10K run
        distance_m = 10000
        duration_s = 3000  # 50 minutes
        pace_s_per_km = duration_s / (distance_m / 1000)  # 300 s/km = 5:00/km
        
        elevations = [10.0, 15.0, 12.0, 20.0]
        elev_gain = elevation_gain(elevations)
        
        assert format_distance(distance_m) == "10.00 km"
        assert format_duration(duration_s) == "0:50:00"
        assert format_pace(pace_s_per_km) == "5:00 /km"
        assert elev_gain == 13.0
    
    def test_safe_pace_calculation(self):
        """Should safely calculate pace even with edge cases"""
        # Normal case
        distance_km = 10.0
        time_s = 3000.0
        pace = safe_divide(time_s, distance_km)
        assert pace == 300.0
        
        # Zero distance (would crash without safe_divide)
        pace_invalid = safe_divide(time_s, 0.0, fallback=0.0)
        assert pace_invalid == 0.0
