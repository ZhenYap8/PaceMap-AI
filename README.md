# 🏃‍♂️ PaceMap AI

**Machine Learning-powered running analytics platform** that processes Strava GPX files to visualize pace heatmaps and predict race finish times.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

PaceMap AI provides two core functionalities:

1. **Single Run Analysis** - Detailed visualization and statistics for individual runs
2. **ML Prediction Model** - Train models on historical data to predict future finish times

### Key Features

- 📊 **Pace visualization** with color-coded heatmaps
- 🗺️ **Interactive maps** using Folium
- 🤖 **3 ML models** (Linear Regression, Random Forest, Gradient Boosting)
- 📈 **Performance metrics** (MAE, RMSE, R²)
- 🔮 **Finish time prediction** for planned routes
- 📉 **Loss curves and diagnostics** for model evaluation

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PaceMap-AI.git
cd PaceMap-AI

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 📊 Analyze a Single Run

```bash
python analyze_single_run.py data/raw_gpx/run_1.gpx
```

**Output:**
- `output/run_1_pace_chart.png` - Pace over distance chart
- `output/run_1_pace_map.html` - Interactive pace heatmap

**Example Output:**
```
================================================================================
ANALYZING RUN: run_1
================================================================================

📂 Loading GPX file...
   ✓ Loaded 3,885 GPS points
   ✓ Start time: 2024-05-17 08:00:00
   ✓ End time: 2024-05-17 09:06:04

================================================================================
SUMMARY STATISTICS
================================================================================
  Distance        : 8.74 km
  Total Time      : 1:06:04
  Average Pace    : 7:33 min/km
  Elevation Gain  : 26.3 m
  Fastest Pace    : 4:35 min/km
  Slowest Pace    : 12:20 min/km
================================================================================
```

#### 🤖 Train ML Model on All Runs

```bash
python train_ml_model.py
```

**Trains on all 44 GPX files** with 80/10/10 split:
- 35 runs for training
- 5 runs for validation (model selection)
- 4 runs for testing (final evaluation)

**Output:**
```
================================================================================
TRAINING 3 MODELS ON 35 RUNS
================================================================================

  linear                    Validation → MAE=   123s  RMSE=   156s  R²=0.9850
  random_forest             Validation → MAE=    98s  RMSE=   124s  R²=0.9920
  gradient_boosting         Validation → MAE=    87s  RMSE=   112s  R²=0.9945

✓ BEST MODEL: GRADIENT_BOOSTING
```

#### 🔮 Make Predictions

```bash
python train_ml_model.py --predict \
  --model models/best_model_gradient_boosting_20260517_173948.pkl \
  --distance 10 \
  --elevation 100 \
  --pace 300
```

**Output:**
```
================================================================================
PREDICTION FOR NEW RUN
================================================================================
  Distance       : 10.00 km
  Elevation Gain : 100.0 m
  Target Pace    : 5:00 min/km
================================================================================
  Predicted Time : 50:23
================================================================================
```

#### 📈 Generate Model Diagnostics

```bash
python visualize_model.py
```

**Generates:**
- `models/loss_curve.png` - Training loss over iterations
- `models/feature_importance.png` - Feature contribution ranking
- `models/scatter_validation.png` - Predicted vs actual (validation)
- `models/scatter_test.png` - Predicted vs actual (test)

---

## 🏗️ Project Architecture

```
PaceMap-AI/
├── analyze_single_run.py     # Single run analysis script
├── train_ml_model.py          # ML model training script
├── visualize_model.py         # Model diagnostics and plots
├── stravaanalyser.ipynb       # Interactive exploration notebook
├── data/
│   ├── raw_gpx/               # Raw GPX files (44 runs)
│   └── processed/             # Processed data
├── models/                    # Saved ML models and plots
├── output/                    # Analysis results
└── src/
    ├── parser.py              # GPX file parsing
    ├── pace_calculator.py     # Distance/pace calculations
    ├── utils.py               # Helper functions
    ├── map_visualizer.py      # Interactive map generation
    ├── data_loader.py         # Batch GPX processing
    └── ml_model.py            # ML pipeline and training
```

---

## 🧠 Machine Learning Details

### What is X (Input Features)?

The model learns from **9 features** (7 original + 2 engineered):

#### Original Features
| Feature | Description | Example | Source |
|---------|-------------|---------|--------|
| `distance_km` | Total distance | 10.5 km | GPX data |
| `elevation_gain_m` | Total climbing | 150 m | GPX data |
| `avg_pace_s_per_km` | Average pace | 330 s/km (5:30/km) | Calculated |
| `avg_heart_rate_bpm` | Average heart rate | 155 bpm | Placeholder* |
| `temperature_c` | Weather temperature | 20°C | Placeholder* |
| `time_of_day_hour` | Start time | 8 (8am) | GPX timestamp |
| `fatigue_score` | Subjective tiredness | 5/10 | Placeholder* |

*Currently placeholders (same value for all runs). Can be enhanced with GPX extensions or external APIs.

#### Engineered Features
| Feature | Formula | Purpose |
|---------|---------|---------|
| `pace_per_elevation` | `avg_pace / (elevation + 1)` | Captures pace efficiency on hills |
| `distance_x_fatigue` | `distance × fatigue` | Models cumulative fatigue effect |

### What is y (Target Variable)?

`finish_time_s` - **Total run duration in seconds**

Extracted from GPX file's start/end timestamps.

### Training Process

```python
# Pseudocode workflow
all_runs = load_all_gpx_files()  # 44 runs
train, val, test = split_data(all_runs, [0.8, 0.1, 0.1])

# Train 3 models
for model in [LinearRegression, RandomForest, GradientBoosting]:
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_val, y_val)

# Select best model by R²
best_model = max(models, key=lambda m: m.r2_score)

# Final evaluation on unseen test set
test_metrics = evaluate(best_model, X_test, y_test)
save_model(best_model)
```

### Model Comparison

| Model | Training Method | Loss Graph? | Interpretability |
|-------|----------------|-------------|------------------|
| **Linear Regression** | Closed-form equation | ❌ No (instant solution) | ✅ High (see coefficients) |
| **Random Forest** | 100 independent trees | ❌ No (built once) | ⚠️ Medium (feature importance) |
| **Gradient Boosting** | 100 sequential trees | ✅ **Yes!** (iterative) | ⚠️ Medium (feature importance) |

**Gradient Boosting** typically performs best due to its ability to capture complex interactions.

### Performance Metrics

- **MAE (Mean Absolute Error)**: Average prediction error in seconds
  - Example: MAE = 87s means predictions are off by ~1.5 minutes on average
  
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more heavily
  - If RMSE >> MAE, model has some outlier predictions
  
- **R² (R-squared)**: How much variance the model explains (0 to 1)
  - R² = 0.9945 means model explains 99.45% of variance
  - Higher is better (1.0 = perfect)

### Why No Epoch Training?

**Linear Regression:**
- Solves `w = (X^T X)^(-1) X^T y` mathematically (no iterations)

**Random Forest:**
- Builds 100 trees independently (no sequential improvement)

**Gradient Boosting:**
- **Does have iterations!** (100 boosting stages)
- Each tree corrects previous errors
- Loss decreases with each iteration (extractable via `visualize_model.py`)

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW GPX FILES (44 runs)                  │
│                   data/raw_gpx/*.gpx                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   parser.py         │  Parse XML → TrackPoints
                │   load_gpx_file()   │  Extract coordinates,
                │   extract_points()  │  timestamps, elevations
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ pace_calculator.py  │  GPS smoothing
                │ calculate_paces()   │  Distance/pace calculations
                │ smooth_gps()        │  Haversine formula
                └──────────┬──────────┘
                           │
           ┌───────────────┴────────────────┐
           │                                │
           ▼                                ▼
  ┌────────────────┐              ┌────────────────┐
  │ SINGLE RUN     │              │ ALL RUNS       │
  │ analyze_single │              │ data_loader.py │
  │ _run.py        │              │ load_all_runs()│
  └────────┬───────┘              └────────┬───────┘
           │                               │
           ▼                               ▼
  ┌────────────────┐              ┌────────────────┐
  │ map_visualizer │              │ ml_model.py    │
  │ build_map()    │              │ train_model()  │
  │ pace_to_color()│              │ predict()      │
  └────────┬───────┘              └────────┬───────┘
           │                               │
           ▼                               ▼
  ┌────────────────┐              ┌────────────────┐
  │ OUTPUT FILES   │              │ SAVED MODELS   │
  │ - pace_chart   │              │ - .pkl files   │
  │ - pace_map.html│              │ - diagnostics  │
  └────────────────┘              └────────────────┘
```

---

## 🔧 Module Breakdown

### [`src/parser.py`](src/parser.py)
**Purpose:** Convert GPX XML files into Python data structures

**Key Functions:**
- `load_gpx_file()` - Parse GPX file from disk
- `extract_track_points()` - Extract GPS points (lat, lon, elevation, time)
- `extract_coordinates()` - Get (lat, lon) tuples
- `extract_timestamps()` - Get datetime objects
- `extract_elevations()` - Get altitude readings

**Data Flow:**
```
GPX File (XML) → gpxpy.parse() → TrackPoint dataclass → Separate lists
```

### [`src/pace_calculator.py`](src/pace_calculator.py)
**Purpose:** Calculate distances and paces from GPS coordinates

**Key Functions:**
- `smooth_gps_coordinates()` - Moving average to reduce GPS noise
- `calculate_segment_distances()` - Haversine formula for distance between points
- `calculate_cumulative_distance()` - Running total distance
- `calculate_segment_paces()` - Time/distance for each segment

**Formula:**
```python
pace (s/km) = time_between_points / distance_between_points
```

### [`src/map_visualizer.py`](src/map_visualizer.py)
**Purpose:** Generate interactive Folium maps with pace heatmaps

**Key Functions:**
- `pace_to_colour()` - Maps pace → hex color (blue=slow, green=moderate, red=fast)
- `create_base_map()` - Creates Folium map centered on route
- `render_pace_route()` - Draws colored line segments
- `add_start_end_markers()` - Green/red markers for start/finish

**Color Scale:**
```
240 s/km (4:00) → Red   (fast)
330 s/km (5:30) → Green (moderate)
420 s/km (7:00) → Blue  (slow)
```

### [`src/data_loader.py`](src/data_loader.py)
**Purpose:** Batch process all GPX files into ML-ready DataFrame

**Key Functions:**
- `extract_run_features()` - Process 1 GPX → 1 feature dictionary
- `load_all_runs()` - Process all GPX files → DataFrame

**Output:**
```python
DataFrame (44 rows × 9 columns):
  run_id | distance_km | elevation_gain_m | avg_pace_s_per_km | finish_time_s | ...
```

### [`src/ml_model.py`](src/ml_model.py)
**Purpose:** ML training pipeline and prediction

**Key Functions:**
- `preprocess_features()` - Clean data, impute missing values
- `engineer_features()` - Create `pace_per_elevation`, `distance_x_fatigue`
- `build_model()` - Create Pipeline (StandardScaler + Model)
- `train_model()` - Full training workflow
- `evaluate_model()` - Calculate MAE, RMSE, R²
- `predict()` - Make prediction for new run

**Pipeline:**
```python
Pipeline([
    ("scaler", StandardScaler()),     # Normalize features
    ("model", GradientBoostingRegressor())  # Train model
])
```

### [`src/utils.py`](src/utils.py)
**Purpose:** Helper functions for formatting and calculations

**Key Functions:**
- `format_pace()` - Seconds → "5:30 /km"
- `format_duration()` - Seconds → "54:07"
- `format_distance()` - Meters → "10.5 km"
- `elevation_gain()` - Sum positive elevation changes
- `elapsed_seconds()` - Time difference between timestamps

---

## 📈 Understanding the Visualizations

### Loss Curve (`models/loss_curve.png`)

Shows how Mean Squared Error decreases with each boosting iteration:

```
MSE
  │
  │  ╲ Training Loss
  │   ╲___________________
  │        ╲ Validation Loss
  │         ╲_____________
  └─────────────────────────► Iteration
    0   20   40   60   80  100
```

**What to look for:**
- Both curves should decrease
- If validation loss increases → **overfitting**
- If both stay high → **underfitting**

### Feature Importance (`models/feature_importance.png`)

Shows which features the model relies on most:

```
distance_km             ████████████ 0.4521
avg_pace_s_per_km       ████████ 0.3201
elevation_gain_m        ████ 0.1532
distance_x_fatigue      ██ 0.0746
pace_per_elevation      █ 0.0312
```

**Interpretation:**
- Distance is the strongest predictor (45%)
- Pace is second most important (32%)
- Heart rate/weather placeholders contribute minimally

### Prediction Scatter (`models/scatter_test.png`)

Predicted vs actual finish times:

```
Predicted
    │      ●
    │    ●   ●
    │  ●   ●
    │●   ●
    └────────────► Actual
```

**Ideal:** Points lie on the diagonal (perfect prediction)

---

## 🎓 Common Questions

### Q: Why is run_6.gpx prediction so bad (104% error)?

**A:** It's likely an **outlier** in your dataset:
- Much shorter/faster than training runs
- Different running pattern (sprint vs. endurance)
- Small dataset (44 runs) limits model's ability to generalize

**Solution:** Collect more diverse runs to train the model better.

### Q: Can I use this with other fitness apps?

**A:** Yes! Any app that exports GPX files:
- Strava ✅
- Garmin Connect ✅
- Runkeeper ✅
- Nike Run Club ✅

### Q: How do I add heart rate data?

**A:** Modify `src/data_loader.py`:
```python
# In extract_run_features():
# Parse <extensions> from GPX
heart_rates = [point.extensions.get('hr') for point in track_points]
avg_hr = np.mean([hr for hr in heart_rates if hr])
features['avg_heart_rate_bpm'] = avg_hr
```

### Q: Can I predict specific routes?

**A:** Yes! Create a GPX file for the planned route, extract its features:
```python
features = extract_run_features('planned_route.gpx')
predicted_time = predict(model, features)
```

---

## 🚧 Roadmap

- [ ] Support for cycling/hiking activities
- [ ] Weather API integration (OpenWeatherMap)
- [ ] Heart rate zone analysis
- [ ] VO2 max estimation
- [ ] Split time predictions (5K, 10K, etc.)
- [ ] Web dashboard with Flask/Streamlit
- [ ] Mobile app integration

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **scikit-learn** - Machine learning framework
- **Folium** - Interactive maps
- **gpxpy** - GPX file parsing
- **matplotlib** - Data visualization

---

## 📧 Contact

Questions? Open an issue or reach out!

**Built with ❤️ for runners by runners** 🏃‍♂️💨
