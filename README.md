# 🏃‍♂️ PaceMap AI

**AI-powered running performance analysis and prediction using GPS data**

[![Tests](https://img.shields.io/badge/tests-195%20passing-brightgreen)](./TEST_STRATEGY.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

PaceMap AI analyses your running data from GPX files to visualise pace variations, predict finish times, and provide insights into your training patterns using machine learning.

![PaceMap AI Demo](https://via.placeholder.com/800x400/4A90E2/FFFFFF?text=PaceMap+AI+Demo)

---

## ✨ Features

### 📊 **Machine Learning Predictions**
- **Finish Time Prediction**: Predict race finish times based on distance, elevation, weather, and fatigue
- **3 ML Models**: Linear Regression, Random Forest, and Gradient Boosting
- **Feature Engineering**: Automatically creates advanced features like pace_per_elevation and distance_x_fatigue
- **Model Evaluation**: MAE, RMSE, and R² metrics with visualisation

### 🗺️ **Interactive Pace Maps**
- **Colour-Coded Routes**: Visualise pace variations with intuitive colour gradients
  - 🔴 Red = Fast pace (< 4:00/km)
  - 🟢 Green = Moderate pace (5:30/km)
  - 🔵 Blue = Slow pace (> 7:00/km)
- **Interactive HTML Maps**: Hover over segments to see detailed pace information
- **Start/Finish Markers**: Clear visual indicators for route endpoints

### 📈 **Performance Analytics**
- **Segment-by-Segment Analysis**: Detailed breakdown of every km/mile
- **Elevation Gain Calculation**: Accurate climbing metrics
- **GPS Smoothing**: Reduces noise from GPS signal variations
- **Pace Charts**: Visual representations of pace over distance

### �� **Data Processing**
- **GPX File Support**: Standard GPS exchange format
- **Batch Processing**: Analyse multiple runs at once
- **Data Validation**: Automatic handling of corrupted/incomplete files
- **CSV Export**: Export processed data for further analysis

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PaceMap-AI.git
   cd PaceMap-AI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1️⃣ **Analyse a Single Run**

```bash
python analyze_single_run.py data/raw_gpx/run_1.gpx
```

This will generate:
- `output/run_1_pace_map.html` - Interactive pace map
- `output/run_1_pace_chart.png` - Pace variation chart
- Console output with run statistics

**Example output:**
```
🏃 ANALYSING RUN: run_1.gpx
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 RUN STATISTICS
  Distance:        10.42 km
  Duration:        0:52:18
  Avg Pace:        5:01 /km
  Elevation Gain:  127 m

✅ Pace map saved to: output/run_1_pace_map.html
✅ Pace chart saved to: output/run_1_pace_chart.png
```

#### 2️⃣ **Train ML Model**

```bash
python train_ml_model.py
```

This will:
- Load all GPX files from `data/raw_gpx/`
- Extract features (distance, elevation, pace, etc.)
- Train 3 different ML models
- Save the best model to `models/`
- Generate performance metrics

**Example output:**
```
📊 MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: Gradient Boosting
  MAE:   89.23 seconds (1:29)
  RMSE:  124.56 seconds (2:05)
  R²:    0.9134

✅ Best model saved: models/best_model_gradient_boosting_20260518.pkl
```

#### 3️⃣ **Visualise Model Performance**

```bash
python visualize_model.py
```

Generates diagnostic visualisations:
- `models/loss_curve.png` - Training/validation loss over iterations
- `models/feature_importance.png` - Which features matter most
- `models/scatter_validation.png` - Predicted vs actual times (validation)
- `models/scatter_test.png` - Predicted vs actual times (test)

---

## 📂 Project Structure

```
PaceMap-AI/
├── analyze_single_run.py      # Single run analysis script
├── train_ml_model.py           # ML model training
├── visualize_model.py          # Model diagnostics
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test configuration
├── TEST_STRATEGY.md            # Testing documentation (195 tests)
│
├── data/
│   ├── raw_gpx/               # Place your GPX files here
│   │   ├── run_1.gpx
│   │   ├── run_2.gpx
│   │   └── ...
│   └── processed/             # Auto-generated CSV files
│
├── src/                       # Core library modules
│   ├── parser.py              # GPX file parsing
│   ├── pace_calculator.py     # Distance/pace calculations
│   ├── map_visualizer.py      # Interactive map generation
│   ├── ml_model.py            # Machine learning models
│   ├── data_loader.py         # Batch data loading
│   └── utils.py               # Utility functions
│
├── tests/                     # Comprehensive test suite (195 tests)
│   ├── test_parser.py         # 16 tests
│   ├── test_ml_model.py       # 27 tests
│   ├── test_utils.py          # 50 tests
│   ├── test_pace_calculator.py # 47 tests
│   ├── test_map_visualizer.py # 33 tests
│   └── test_data_loader.py    # 23 tests
│
├── models/                    # Trained ML models
│   ├── best_model_*.pkl
│   └── *.png                  # Visualisation outputs
│
└── output/                    # Generated maps and charts
    ├── *_pace_map.html
    └── *_pace_chart.png
```

---

## 🧪 Testing

PaceMap AI has a **world-class test suite** with 195 tests covering all modules.

### Run All Tests

```bash
pytest tests/ -v
```

**Expected output:**
```
195 passed in 1.36s ✅
```

### Run Specific Test Modules

```bash
pytest tests/test_parser.py -v              # GPX parsing (16 tests)
pytest tests/test_ml_model.py -v            # ML models (27 tests)
pytest tests/test_utils.py -v               # Utilities (50 tests)
pytest tests/test_pace_calculator.py -v     # Calculations (47 tests)
pytest tests/test_map_visualizer.py -v      # Visualisation (33 tests)
pytest tests/test_data_loader.py -v         # Data loading (23 tests)
```

### Generate Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Current coverage: 96%+ across all modules**

For detailed testing documentation, see [TEST_STRATEGY.md](./TEST_STRATEGY.md).

---

## 📊 Understanding the Outputs

### **Scatter Test Plot** (`scatter_test.png`)

The scatter test plot shows how well the ML model predicts finish times on unseen data:

**What to look for:**
- **Points near the diagonal line** = Accurate predictions ✅
- **Points above the line** = Model overestimated (predicted slower than actual)
- **Points below the line** = Model underestimated (predicted faster than actual)
- **Colour intensity** = Prediction error magnitude
  - Green = Low error (good prediction)
  - Yellow = Moderate error
  - Red = High error (poor prediction)

**Interpreting the metrics:**
- **MAE (Mean Absolute Error)**: Average prediction error in seconds
  - Example: MAE = 89s means predictions are off by ~1.5 minutes on average
- **Perfect diagonal line**: Where predicted = actual (ideal model)
- **Tight clustering around line**: Model is reliable and consistent

**Example interpretation:**
```
If most points are within ±2 minutes of the diagonal line,
your model is production-ready for race planning!
```

### **Feature Importance** (`feature_importance.png`)

Shows which factors most influence your finish time:

**Typical importance ranking:**
1. 🥇 **Distance** (40-50%) - Most important factor
2. 🥈 **Elevation gain** (20-30%) - Second most important
3. 🥉 **Average pace** (15-20%) - Historical pace patterns
4. **Fatigue score** (5-10%) - Training load impact
5. **Temperature** (2-5%) - Weather conditions

### **Loss Curve** (`loss_curve.png`)

Shows how the model improves during training:

- **Decreasing curves** = Model is learning ✅
- **Training/validation gap** = Potential overfitting if large
- **Convergence point** = Optimal number of training iterations

---

## 🛠️ Advanced Usage

### Custom ML Model Training

```bash
# Train specific model type
python train_ml_model.py --model-type gradient_boosting

# Use custom data directory
python train_ml_model.py --gpx-dir /path/to/gpx/files

# Adjust train/test split
python train_ml_model.py --test-size 0.3
```

### Batch Analysis

Process multiple runs and export to CSV:

```python
from src.data_loader import load_all_runs

# Load all runs from directory
df = load_all_runs('data/raw_gpx/')

# Export to CSV
df.to_csv('data/processed/all_runs.csv', index=False)

print(f"Processed {len(df)} runs")
```

### Custom Pace Map

```python
from src.parser import load_gpx_file, extract_track_points, extract_coordinates
from src.pace_calculator import calculate_segment_paces, smooth_gps_coordinates
from src.map_visualizer import build_activity_map, export_map_html

# Parse GPX
gpx = load_gpx_file('my_run.gpx')
points = extract_track_points(gpx)
coords = extract_coordinates(points)

# Calculate paces
coords_smooth = smooth_gps_coordinates(coords, window=3)
timestamps = extract_timestamps(points)
paces = calculate_segment_paces(coords_smooth, timestamps)

# Create map
activity_map = build_activity_map(coords, paces, "My Run", zoom_start=15)
export_map_html(activity_map, 'output/my_custom_map.html')
```

---

## 🎯 Use Cases

### 1. **Race Time Prediction**
Input your target race distance and typical training conditions to get a predicted finish time.

### 2. **Training Analysis**
Identify pace trends, elevation impact, and improvement over time across multiple training runs.

### 3. **Route Planning**
Visualise and compare different running routes to understand pace variations on terrain.

### 4. **Performance Tracking**
Monitor how environmental factors (temperature, elevation, fatigue) affect your performance.

### 5. **Pacing Strategy**
Analyse historical data to develop optimal pacing strategies for races.

---

## 🔬 Technical Details

### ML Features

**Input features for prediction:**
- `distance_km` - Route distance
- `elevation_gain_m` - Total climbing
- `avg_pace_s_per_km` - Historical average pace
- `avg_heart_rate_bpm` - Average heart rate (if available)
- `temperature_c` - Weather conditions
- `time_of_day_hour` - Start time (circadian rhythm effects)
- `fatigue_score` - Training load/recovery status

**Engineered features:**
- `pace_per_elevation` - Efficiency on hills
- `distance_x_fatigue` - Combined endurance/fatigue effect

### Pace Calculation

Uses the **Haversine formula** for accurate GPS distance calculation:

```python
distance = 2 * R * arcsin(√(sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)))
```

Where R = 6371 km (Earth's radius)

### GPS Smoothing

Applies **moving average filter** with configurable window size to reduce GPS noise:

```python
smoothed_coords = moving_average(coordinates, window=3)
```

---

## 📦 Dependencies

Core dependencies:
- **gpxpy** - GPX file parsing
- **folium** - Interactive map generation
- **scikit-learn** - Machine learning models
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **matplotlib** - Chart generation

Development dependencies:
- **pytest** - Testing framework (195 tests)
- **pytest-cov** - Coverage reporting

See [requirements.txt](./requirements.txt) for full list with versions.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   pytest tests/ -v
   ```
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- Write tests for new features (see [TEST_STRATEGY.md](./TEST_STRATEGY.md))
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Update README if adding new functionality
- Ensure all 195 tests pass before submitting PR

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "No module named 'gpxpy'"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue: "Cannot create a map with no coordinates"**
```bash
# Solution: Check GPX file has valid track points
python -c "import gpxpy; gpx = gpxpy.parse(open('your_file.gpx'))"
```

**Issue: "Insufficient data for training"**
```bash
# Solution: You need at least 10 GPX files for ML training
# Add more runs to data/raw_gpx/
```

**Issue: Model predictions seem inaccurate**
```bash
# Solution: Retrain with more data or try different model
python train_ml_model.py --model-type gradient_boosting
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Folium](https://python-visualization.github.io/folium/) for interactive maps
- Inspired by Strava and Garmin Connect analytics
- GPS parsing powered by [gpxpy](https://github.com/tkrajina/gpxpy)
- Machine learning with [scikit-learn](https://scikit-learn.org/)

---

## 📧 Contact

**Project Link:** [https://github.com/yourusername/PaceMap-AI](https://github.com/yourusername/PaceMap-AI)

**Issues:** [https://github.com/yourusername/PaceMap-AI/issues](https://github.com/yourusername/PaceMap-AI/issues)

---

## 🗺️ Roadmap

### v2.0 (Planned)
- [ ] Heart rate zone analysis
- [ ] Weather API integration (automatic temperature/conditions)
- [ ] Strava API integration
- [ ] Web dashboard (Flask/FastAPI)
- [ ] Mobile app companion

### v1.5 (In Progress)
- [ ] Real-time race predictor
- [ ] Interval training analysis
- [ ] Comparative run analysis
- [ ] VO2max estimation

### v1.0 (Current) ✅
- [x] GPX parsing and analysis
- [x] ML-based finish time prediction
- [x] Interactive pace maps
- [x] Comprehensive test suite (195 tests)
- [x] Batch processing
- [x] Model visualisation

---

<p align="center">
  Made with ❤️ for runners by runners
</p>

<p align="center">
  <strong>⭐ Star this repo if you find it useful!</strong>
</p>
