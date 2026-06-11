# 🧪 PaceMap AI Testing Strategy

## Overview

Comprehensive test suite following software engineering best practices with **42 tests** covering critical components of the ML pipeline.

**Test Coverage:**
- ✅ **42/42 tests passing (100%)**
- 🎯 **2 test files** (parser, ml_model)
- 📦 **6 test classes** with unit and integration tests
- ⚡ **Execution time:** ~1.3 seconds

---

## Test Structure

```
tests/
├── __init__.py              # Test package marker
├── test_parser.py           # GPX parsing tests (15 tests)
├── test_ml_model.py         # ML model tests (27 tests)
└── fixtures/                # Test data (future)
```

---

## Test Files

### 1. `test_parser.py` (15 tests)

Tests GPX file parsing and data extraction.

#### Test Classes:

**`TestLoadGPXFile` (4 tests)**
- ✅ Load valid GPX file
- ✅ Raise error for nonexistent file
- ✅ Handle empty file gracefully
- ✅ Raise error for corrupted XML

**`TestExtractTrackPoints` (3 tests)**
- ✅ Extract track points from valid GPX
- ✅ Return empty list for GPX with no tracks
- ✅ Handle multiple track segments

**`TestExtractCoordinates` (2 tests)**
- ✅ Extract (lat, lon) tuples
- ✅ Return empty list for no points

**`TestExtractTimestamps` (3 tests)**
- ✅ Extract datetime objects
- ✅ Return empty list for no points
- ✅ Handle points with missing timestamps

**`TestExtractElevations` (3 tests)**
- ✅ Extract elevation values
- ✅ Return empty list for no points
- ✅ Handle points without elevation data

**`TestGPXParsingWorkflow` (1 integration test)**
- ✅ Complete parsing pipeline (GPX → coordinates, timestamps, elevations)

---

### 2. `test_ml_model.py` (27 tests)

Tests ML preprocessing, training, evaluation, and prediction.

#### Test Classes:

**`TestPreprocessFeatures` (5 tests)**
- ✅ Preprocess valid data
- ✅ Drop rows with missing target values
- ✅ Impute missing features with median
- ✅ Raise error for missing required columns
- ✅ Raise error if target column missing

**`TestEngineerFeatures` (4 tests)**
- ✅ Create pace_per_elevation and distance_x_fatigue
- ✅ Calculate pace_per_elevation correctly
- ✅ Calculate distance_x_fatigue correctly
- ✅ Handle zero elevation without division error

**`TestBuildModel` (4 tests)**
- ✅ Build linear regression pipeline
- ✅ Build random forest pipeline
- ✅ Build gradient boosting pipeline
- ✅ Raise error for invalid model type

**`TestTrainModel` (4 tests)**
- ✅ Train on valid data
- ✅ Produce reproducible train/test split (random_state=42)
- ✅ Successfully train all 3 model types
- ✅ Handle insufficient data gracefully

**`TestEvaluateModel` (3 tests)**
- ✅ Return MAE, RMSE, R² metrics
- ✅ Produce reasonable metric values
- ✅ Verify RMSE ≥ MAE (mathematical property)

**`TestPredict` (4 tests)**
- ✅ Make prediction for single run
- ✅ Return numeric finish time (no NaN/inf)
- ✅ Raise error for incomplete features
- ✅ Handle invalid feature values appropriately

**`TestModelIntegration` (2 integration tests)**
- ✅ Complete ML pipeline (train → evaluate → predict)
- ✅ Consistent predictions for same input

---

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_parser.py -v
pytest tests/test_ml_model.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_ml_model.py::TestPreprocessFeatures -v
```

### Run Single Test
```bash
pytest tests/test_ml_model.py::TestPredict::test_predict_single_run -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=pacemap --cov-report=html
```

### Run in Quiet Mode
```bash
pytest tests/ -q
```

---

## Test Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
markers =
    integration: Integration tests
    slow: Slow tests
```

---

## Testing Best Practices Applied

### 1. **AAA Pattern (Arrange-Act-Assert)**
```python
def test_preprocess_valid_data(self, sample_data):
    # Arrange
    data = sample_data
    
    # Act
    cleaned = preprocess_features(data)
    
    # Assert
    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) == len(data)
```

### 2. **Fixtures for Test Data**
```python
@pytest.fixture
def sample_data():
    """Generate valid sample training data"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({...})
```

### 3. **Edge Case Testing**
- Empty inputs
- Missing/NaN values
- Invalid model types
- Corrupted files
- Negative values

### 4. **Error Testing**
```python
def test_raise_error_invalid_model(self):
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model('invalid_model')
```

### 5. **Integration Tests**
```python
def test_full_ml_pipeline(self, sample_data):
    # Tests complete workflow: load → train → evaluate → predict
    pipeline, X_train, y_train, X_test, y_test = train_model(...)
    metrics = evaluate_model(...)
    prediction = predict(...)
    assert all components work together
```

### 6. **Reproducibility**
```python
def test_reproducible_split(self, sample_data):
    # Same random_state should produce same split
    _, X_train1, _, _, _ = train_model(sample_data, random_state=42)
    _, X_train2, _, _, _ = train_model(sample_data, random_state=42)
    assert X_train1.index.tolist() == X_train2.index.tolist()
```

---

## Test Coverage by Module

| Module | Tests | Coverage |
|--------|-------|----------|
| `parser.py` | 15 | ✅ High (core functions) |
| `ml_model.py` | 27 | ✅ High (all functions) |
| `pace_calculator.py` | 0 | ⚠️ Not yet covered |
| `map_visualizer.py` | 0 | ⚠️ Not yet covered |
| `data_loader.py` | 0 | ⚠️ Not yet covered |
| `utils.py` | 0 | ⚠️ Not yet covered |

---

## Future Test Additions

### Priority 1: Core Calculations
```python
# tests/test_pace_calculator.py
- test_calculate_segment_distances()
- test_calculate_cumulative_distance()
- test_calculate_segment_paces()
- test_smooth_gps_coordinates()
```

### Priority 2: Data Loading
```python
# tests/test_data_loader.py
- test_extract_run_features()
- test_load_all_runs()
- test_handle_corrupted_gpx_in_batch()
```

### Priority 3: Utilities
```python
# tests/test_utils.py
- test_format_pace()
- test_format_duration()
- test_elevation_gain()
- test_haversine_distance()
```

### Priority 4: Visualization
```python
# tests/test_map_visualizer.py
- test_pace_to_colour()
- test_create_base_map()
- test_render_pace_route()
```

---

## Continuous Integration (Future)

### GitHub Actions Workflow
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --cov=pacemap
```

---

## Test Maintenance

### When to Update Tests

1. **Adding new features** → Write tests first (TDD)
2. **Fixing bugs** → Add regression test
3. **Refactoring** → Ensure tests still pass
4. **Changing APIs** → Update affected tests

### Test Quality Checklist

- [ ] Test names clearly describe what they test
- [ ] Each test tests one thing
- [ ] Tests are independent (no shared state)
- [ ] Tests use fixtures for setup
- [ ] Edge cases are covered
- [ ] Error conditions are tested
- [ ] Integration tests cover workflows

---

## Bug Fixes During Testing

### Issue 1: Pandas Copy-on-Write Warning
**Problem:** `df[col].fillna(median_val, inplace=True)` triggered warning  
**Fix:** Changed to `df[col] = df[col].fillna(median_val)`  
**Test:** `test_impute_missing_features`

### Issue 2: GPXTrackPoint Attribute Error
**Problem:** Test used wrong attribute name (`.timestamp` vs `.time`)  
**Fix:** Used internal `TrackPoint` dataclass instead  
**Test:** `test_handle_missing_timestamps`

---

## Metrics

- **Total Tests:** 42
- **Pass Rate:** 100%
- **Execution Time:** 1.34 seconds
- **Files Tested:** 2/6 modules
- **Test-to-Code Ratio:** ~1:2 (good coverage for tested modules)

---

## Contributing Tests

When adding new tests:

1. Follow existing test structure
2. Use descriptive test names: `test_<function>_<scenario>`
3. Add docstrings explaining what's being tested
4. Use fixtures for reusable test data
5. Test both happy path and edge cases
6. Run full test suite before committing

```bash
# Run tests before committing
pytest tests/ -v

# Check for any failures
echo $?  # Should output 0 for success
```

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Python Testing with pytest (Book)](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)

---

**Last Updated:** 2026-05-17  
**Test Suite Version:** 1.0  
**Maintained by:** PaceMap AI Team
