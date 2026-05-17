"""
analyze_single_run.py

Analyzes a single GPX run and provides:
- Summary statistics
- Pace over distance chart
- Interactive pace heatmap
- Elevation profile

Usage:
    python analyze_single_run.py data/raw_gpx/run_1.gpx
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from parser import load_gpx_file, extract_track_points, extract_coordinates, extract_timestamps, extract_elevations
from pace_calculator import smooth_gps_coordinates, calculate_segment_distances, calculate_cumulative_distance, calculate_segment_paces
from utils import format_pace, format_distance, format_duration, elevation_gain, elapsed_seconds, metres_to_km, deduplicate_timestamps
from map_visualizer import build_activity_map, export_map_html


def analyze_run(gpx_filepath: str, output_dir: str = "output", smoothing_window: int = 3):
    """
    Analyze a single GPX run and generate visualizations.
    
    Args:
        gpx_filepath: Path to the GPX file
        output_dir: Directory to save output files
        smoothing_window: Window size for GPS smoothing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    run_name = os.path.splitext(os.path.basename(gpx_filepath))[0]
    print(f"\n{'='*80}")
    print(f"ANALYZING RUN: {run_name}")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # STEP 1: Parse GPX File
    # =========================================================================
    print("📂 Loading GPX file...")
    gpx = load_gpx_file(gpx_filepath)
    track_points = extract_track_points(gpx)
    coordinates = extract_coordinates(track_points)
    timestamps = extract_timestamps(track_points)
    elevations = extract_elevations(track_points)
    
    print(f"   ✓ Loaded {len(track_points)} GPS points")
    print(f"   ✓ Start time: {timestamps[0]}")
    print(f"   ✓ End time: {timestamps[-1]}")
    
    # =========================================================================
    # STEP 2: Calculate Metrics
    # =========================================================================
    print("\n📊 Calculating metrics...")
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
    
    # =========================================================================
    # STEP 3: Display Summary Statistics
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"  Distance        : {format_distance(total_distance_m)}")
    print(f"  Total Time      : {format_duration(total_elapsed or 0)}")
    print(f"  Average Pace    : {format_pace(avg_pace)}")
    print(f"  Elevation Gain  : {elev_gain:.1f} m")
    print(f"  GPS Points      : {len(track_points)}")
    print(f"  Segments        : {len(segment_paces)}")
    
    # Pace statistics
    if valid_paces:
        fastest_pace = min(valid_paces)
        slowest_pace = max(valid_paces)
        print(f"  Fastest Pace    : {format_pace(fastest_pace)}")
        print(f"  Slowest Pace    : {format_pace(slowest_pace)}")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # STEP 4: Generate Pace Over Distance Chart
    # =========================================================================
    print("📈 Generating pace chart...")
    
    cum_km = [d / 1000 for d in cumulative[:-1]]
    pace_mins = [p / 60 for p in segment_paces]
    filtered = [(k, p) for k, p in zip(cum_km, pace_mins) if 0 < p < 20]
    x_vals, y_vals = zip(*filtered) if filtered else ([], [])
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x_vals, y_vals, linewidth=1.5, color='steelblue', alpha=0.8, label='Pace')
    ax.fill_between(x_vals, y_vals, alpha=0.2, color='steelblue')
    ax.axhline(avg_pace / 60, color='tomato', linestyle='--', linewidth=2, label=f'Avg {format_pace(avg_pace)}')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}:{int((v % 1)*60):02d}"))
    ax.invert_yaxis()
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Pace (min/km)', fontsize=12)
    ax.set_title(f'Pace Over Distance - {run_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, f"{run_name}_pace_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved pace chart: {chart_path}")
    plt.close()
    
    # =========================================================================
    # STEP 5: Generate Interactive Pace Map
    # =========================================================================
    print("🗺️  Generating interactive map...")
    
    activity_map = build_activity_map(
        coordinates=smooth_coords,
        paces=segment_paces,
        activity_name=run_name,
        zoom_start=14,
    )
    
    map_path = os.path.join(output_dir, f"{run_name}_pace_map.html")
    export_map_html(activity_map, map_path)
    print(f"   ✓ Saved interactive map: {map_path}")
    
    # =========================================================================
    # STEP 6: Summary Report
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output files saved to: {output_dir}/")
    print(f"  - {run_name}_pace_chart.png")
    print(f"  - {run_name}_pace_map.html")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single GPX run with visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_single_run.py data/raw_gpx/run_1.gpx
  python analyze_single_run.py data/raw_gpx/run_5.gpx --output results
  python analyze_single_run.py data/raw_gpx/run_10.gpx --smoothing 5
        """
    )
    
    parser.add_argument(
        'gpx_file',
        type=str,
        help='Path to the GPX file to analyze'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for generated files (default: output)'
    )
    
    parser.add_argument(
        '--smoothing', '-s',
        type=int,
        default=3,
        help='GPS smoothing window size (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.gpx_file):
        print(f"❌ Error: GPX file not found: {args.gpx_file}")
        sys.exit(1)
    
    # Run analysis
    analyze_run(args.gpx_file, args.output, args.smoothing)


if __name__ == "__main__":
    main()
