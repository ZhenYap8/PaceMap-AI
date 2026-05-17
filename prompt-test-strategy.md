# Agent Task: Edge Cases and Testing Strategy for PaceMap AI

## Objective

Improve the robustness of PaceMap AI by implementing proper edge case handling and a clear testing strategy across the GPX parsing, pace calculation, mapping, and machine learning pipeline.

This project processes Strava GPX files, extracts GPS coordinates, timestamps, elevation data, calculates pace metrics, visualizes runs, and trains ML models to predict finish time.

---

## Priority Areas

Focus on these modules:

```bash
src/parser.py
src/pace_calculator.py
src/map_visualizer.py
src/data_loader.py
src/ml_model.py
src/utils.py
```

---

## Edge Cases to Handle

### 1. GPX Parsing Edge Cases

Handle:

- Empty GPX files
- Corrupted GPX/XML files
- GPX files with no tracks
- GPX files with no track segments
- Track points missing latitude or longitude
- Track points missing timestamps
- Track points missing elevation
- Multiple tracks in one GPX file
- Multiple segments in one track

Expected behavior:

- Raise clear custom errors where appropriate
- Skip invalid points only when safe
- Never fail silently
- Return useful messages for debugging

---

### 2. Pace Calculation Edge Cases

Handle:

- Duplicate timestamps
- Zero time difference between points
- Zero distance between points
- Stationary GPS drift
- Unrealistic GPS jumps
- Negative or impossible pace values
- Very short runs
- Runs with fewer than 2 valid points

Expected behavior:

- Prevent divide-by-zero errors
- Filter or flag invalid segments
- Use sensible defaults only when documented
- Keep calculations deterministic

---

### 3. Elevation Edge Cases

Handle:

- Missing elevation values
- Large elevation spikes
- Negative elevation values
- No elevation gain
- Noisy elevation readings

Expected behavior:

- Ignore missing elevation safely
- Smooth elevation if needed
- Calculate elevation gain only from positive changes

---

### 4. Map Visualization Edge Cases

Handle:

- No valid coordinates
- Only one coordinate
- Missing pace values
- Extreme pace values
- Output directory missing

Expected behavior:

- Create output directories automatically
- Avoid generating broken maps
- Use fallback colors for missing pace
- Add clear error messages

---

### 5. Machine Learning Edge Cases

Handle:

- Too few GPX files for train/validation/test split
- Missing feature columns
- NaN or infinite values
- Constant placeholder features
- Outlier runs
- Very small datasets
- Model file missing during prediction
- Invalid prediction inputs

Expected behavior:

- Validate dataset before training
- Impute or remove invalid values explicitly
- Warn when dataset is too small
- Ensure train/validation/test split is reproducible
- Save model metadata with features used

---

## Testing Strategy

Create a `tests/` directory:

```bash
tests/
├── test_parser.py
├── test_pace_calculator.py
├── test_utils.py
├── test_data_loader.py
├── test_ml_model.py
└── fixtures/
    ├── valid_run.gpx
    ├── empty.gpx
    ├── corrupted.gpx
    ├── missing_timestamps.gpx
    ├── duplicate_timestamps.gpx
    ├── missing_elevation.gpx
    └── gps_jump.gpx
```

Use `pytest`.

---

## Required Tests

### Parser Tests

Test that:

- valid GPX files load correctly
- corrupted GPX files raise errors
- missing timestamps are handled
- missing elevation does not crash parsing
- coordinates are extracted correctly
- multiple segments are handled

---

### Pace Calculator Tests

Test that:

- Haversine distance is approximately correct
- cumulative distance increases correctly
- duplicate timestamps do not cause division errors
- zero distance segments are ignored or handled
- unrealistic GPS jumps are flagged or filtered
- pace values are calculated in seconds per kilometre

---

### Utility Tests

Test that:

- duration formatting works
- pace formatting works
- elevation gain is calculated correctly
- elapsed time calculations are correct
- invalid inputs are handled clearly

---

### Data Loader Tests

Test that:

- multiple GPX files are loaded into a DataFrame
- missing files are skipped or reported
- required ML columns exist
- feature engineering creates expected columns
- invalid runs do not corrupt the whole dataset

---

### ML Model Tests

Test that:

- preprocessing handles NaN values
- train/validation/test split is reproducible
- models train without crashing on valid data
- predictions return numeric finish times
- invalid prediction inputs raise clear errors
- saved models can be loaded again

---

## Implementation Rules

When modifying the codebase:

1. Read the existing module before editing.
2. Make the smallest safe change.
3. Preserve current functionality.
4. Add tests for every new edge case.
5. Prefer explicit validation over silent assumptions.
6. Use clear error messages.
7. Keep functions modular and typed.

---

## Recommended Engineering Improvements

Add:

- custom exception classes
- input validation functions
- logging instead of print statements where appropriate
- deterministic random seeds for ML
- model metadata saving
- test fixtures for GPX edge cases
- CI-ready pytest command

Example:

```bash
pytest tests/ -v
```

---

## Definition of Done

The task is complete when:

- all core modules have edge case handling
- tests cover parsing, pace calculation, utilities, data loading, and ML
- invalid GPX files do not crash the full pipeline
- ML training validates input data before fitting
- prediction fails gracefully on invalid input
- all tests pass with `pytest`