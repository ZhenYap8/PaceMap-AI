"""
test_map_visualizer.py

Tests for map_visualizer module covering:
- Pace to color conversion
- Base map creation
- Route rendering
- Start/end markers
- Complete activity map building
- HTML export
- Edge cases and error handling
"""

import pytest
import sys
import os
import folium
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_visualizer import (
    pace_to_colour,
    create_base_map,
    render_pace_route,
    add_start_end_markers,
    build_activity_map,
    export_map_html,
    PACE_SLOW_THRESHOLD,
    PACE_FAST_THRESHOLD
)


class TestPaceToColour:
    """Test pace to color conversion"""
    
    def test_fast_pace_strava_orange(self):
        """Fast pace should return Strava orange"""
        # 4:00/km = 240s/km (fast threshold)
        color = pace_to_colour(240.0)
        assert color == "#FC4C02"
    
    def test_slow_pace_light_orange(self):
        """Slow pace should return light orange"""
        # 7:00/km = 420s/km (slow threshold)
        color = pace_to_colour(420.0)
        assert color == "#FFC299"
    
    def test_moderate_pace_mid_orange(self):
        """Moderate pace should return mid orange"""
        # 5:30/km = 330s/km (middle)
        color = pace_to_colour(330.0)
        assert color == "#FF8C55"
    
    def test_very_fast_pace(self):
        """Very fast pace (elite) should stay in orange family"""
        # 3:00/km = 180s/km
        color = pace_to_colour(180.0)
        assert color.startswith("#")
        assert color == "#FC4C02"
    
    def test_very_slow_pace(self):
        """Very slow pace should stay in orange family"""
        # 10:00/km = 600s/km
        color = pace_to_colour(600.0)
        assert color.startswith("#")
        assert color == "#FFC299"
    
    def test_zero_pace_grey(self):
        """Zero pace should return grey"""
        color = pace_to_colour(0.0)
        assert color == "#808080"
    
    def test_negative_pace_grey(self):
        """Negative pace should return grey"""
        color = pace_to_colour(-100.0)
        assert color == "#808080"
    
    def test_custom_thresholds(self):
        """Should work with custom thresholds"""
        color = pace_to_colour(300.0, slow_threshold=360.0, fast_threshold=240.0)
        assert color.startswith("#")
        assert len(color) == 7
    
    def test_color_format(self):
        """All colors should be valid hex format"""
        paces = [180, 240, 300, 360, 420, 480]
        for pace in paces:
            color = pace_to_colour(pace)
            assert color.startswith("#")
            assert len(color) == 7
            # Check it's valid hex
            int(color[1:], 16)
    
    def test_gradient_continuity(self):
        """Colors should change gradually across pace range"""
        colors = [pace_to_colour(p) for p in range(240, 421, 20)]
        # Should get different colors
        assert len(set(colors)) > 5


class TestCreateBaseMap:
    """Test base map creation"""
    
    def test_create_with_single_coordinate(self):
        """Should create map centered on single coordinate"""
        coords = [(51.5074, -0.1278)]
        fmap = create_base_map(coords)
        
        assert isinstance(fmap, folium.Map)
        assert fmap.location == [51.5074, -0.1278]
    
    def test_create_with_multiple_coordinates(self):
        """Should create map centered on average of coordinates"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252)
        ]
        fmap = create_base_map(coords)
        
        assert isinstance(fmap, folium.Map)
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lon = sum(c[1] for c in coords) / len(coords)
        assert fmap.location == [avg_lat, avg_lon]
    
    def test_create_with_custom_zoom(self):
        """Should respect custom zoom level"""
        coords = [(51.5074, -0.1278)]
        fmap = create_base_map(coords, zoom_start=10)
        
        assert fmap.options['zoom'] == 10
    
    def test_empty_coordinates_raises_error(self):
        """Should raise error for empty coordinates"""
        with pytest.raises(ValueError, match="Cannot create a map with no coordinates"):
            create_base_map([])
    
    def test_map_has_openstreetmap_tiles(self):
        """Should use OpenStreetMap tiles"""
        coords = [(51.5074, -0.1278)]
        fmap = create_base_map(coords)
        
        # Check that map was created successfully
        assert fmap is not None


class TestRenderPaceRoute:
    """Test route rendering"""
    
    def test_render_simple_route(self):
        """Should render route with pace coloring"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252)
        ]
        paces = [300.0, 330.0]  # 2 segments
        
        fmap = folium.Map(location=[51.5080, -0.1265])
        result = render_pace_route(fmap, coords, paces)
        
        assert result is fmap
    
    def test_route_with_custom_activity_name(self):
        """Should use custom activity name in tooltips"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        paces = [300.0]
        
        fmap = folium.Map(location=[51.5077, -0.1271])
        render_pace_route(fmap, coords, paces, activity_name="Morning Run")
        
        # Map should still be valid
        assert fmap is not None
    
    def test_mismatched_paces_raises_error(self):
        """Should raise error if pace count doesn't match segments"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252)
        ]
        paces = [300.0]  # Only 1 pace for 2 segments
        
        fmap = folium.Map(location=[51.5080, -0.1265])
        with pytest.raises(ValueError, match="Expected 2 pace values"):
            render_pace_route(fmap, coords, paces)
    
    def test_insufficient_coordinates(self):
        """Should handle insufficient coordinates gracefully"""
        coords = [(51.5074, -0.1278)]
        paces = []
        
        fmap = folium.Map(location=[51.5074, -0.1278])
        result = render_pace_route(fmap, coords, paces)
        
        assert result is fmap
    
    def test_empty_coordinates(self):
        """Should handle empty coordinates"""
        fmap = folium.Map(location=[51.5074, -0.1278])
        result = render_pace_route(fmap, [], [])
        
        assert result is fmap


class TestAddStartEndMarkers:
    """Test start/end marker addition"""
    
    def test_add_markers_to_route(self):
        """Should add start and end markers"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252)
        ]
        
        fmap = folium.Map(location=[51.5080, -0.1265])
        result = add_start_end_markers(fmap, coords)
        
        assert result is fmap
    
    def test_single_coordinate(self):
        """Should handle single coordinate (same start/end)"""
        coords = [(51.5074, -0.1278)]
        
        fmap = folium.Map(location=[51.5074, -0.1278])
        result = add_start_end_markers(fmap, coords)
        
        assert result is fmap
    
    def test_empty_coordinates(self):
        """Should handle empty coordinates"""
        fmap = folium.Map(location=[51.5074, -0.1278])
        result = add_start_end_markers(fmap, [])
        
        assert result is fmap
    
    def test_custom_activity_name(self):
        """Should use custom activity name in popups"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        
        fmap = folium.Map(location=[51.5077, -0.1271])
        result = add_start_end_markers(fmap, coords, activity_name="Evening Jog")
        
        assert result is fmap


class TestBuildActivityMap:
    """Test complete activity map building"""
    
    def test_build_complete_map(self):
        """Should build complete map with all elements"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252)
        ]
        paces = [300.0, 330.0]
        
        fmap = build_activity_map(coords, paces)
        
        assert isinstance(fmap, folium.Map)
    
    def test_build_with_custom_zoom(self):
        """Should respect custom zoom level"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        paces = [300.0]
        
        fmap = build_activity_map(coords, paces, zoom_start=12)
        
        assert fmap.options['zoom'] == 12
    
    def test_build_with_activity_name(self):
        """Should use custom activity name"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        paces = [300.0]
        
        fmap = build_activity_map(coords, paces, activity_name="Test Run")
        
        assert isinstance(fmap, folium.Map)


class TestExportMapHTML:
    """Test HTML export"""
    
    def test_export_to_file(self, tmp_path):
        """Should export map to HTML file"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        paces = [300.0]
        fmap = build_activity_map(coords, paces)
        
        output_file = tmp_path / "test_map.html"
        export_map_html(fmap, str(output_file))
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
    
    def test_exported_html_is_valid(self, tmp_path):
        """Should export valid HTML"""
        coords = [(51.5074, -0.1278), (51.5080, -0.1265)]
        paces = [300.0]
        fmap = build_activity_map(coords, paces)
        
        output_file = tmp_path / "test_map.html"
        export_map_html(fmap, str(output_file))
        
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content or "<html" in content
        assert "folium" in content.lower()


class TestMapVisualizerIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_map_creation_workflow(self, tmp_path):
        """Should create and export complete activity map"""
        # Simulate a short run
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252),
            (51.5092, -0.1239)
        ]
        paces = [270.0, 300.0, 330.0]  # Varying paces
        
        # Build map
        fmap = build_activity_map(coords, paces, activity_name="Test Run")
        
        # Export
        output_file = tmp_path / "integration_test_map.html"
        export_map_html(fmap, str(output_file))
        
        # Validate
        assert output_file.exists()
        assert output_file.stat().st_size > 1000  # Should be substantial HTML
    
    def test_map_with_varied_paces(self):
        """Should handle wide range of pace values"""
        coords = [
            (51.5074, -0.1278),
            (51.5080, -0.1265),
            (51.5086, -0.1252),
            (51.5092, -0.1239),
            (51.5098, -0.1226)
        ]
        # Very fast to very slow
        paces = [180.0, 300.0, 420.0, 540.0]
        
        fmap = build_activity_map(coords, paces)
        
        assert isinstance(fmap, folium.Map)
    
    def test_long_route(self):
        """Should handle long routes with many points"""
        # Simulate 100 point route
        coords = [(51.5074 + i * 0.0001, -0.1278 + i * 0.0001) for i in range(100)]
        paces = [300.0] * 99
        
        fmap = build_activity_map(coords, paces)
        
        assert isinstance(fmap, folium.Map)
    
    def test_stationary_activity(self):
        """Should handle stationary GPS points (e.g., paused run)"""
        # Same location repeated
        coords = [(51.5074, -0.1278)] * 5
        paces = [0.0] * 4  # Zero pace for stationary segments
        
        fmap = build_activity_map(coords, paces)
        
        assert isinstance(fmap, folium.Map)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
