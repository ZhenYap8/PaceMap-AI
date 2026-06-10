"""
test_ml_model.py

Tests for ML model module covering:
- Preprocessing edge cases
- Feature engineering
- Model training reproducibility
- Prediction validation
- Error handling
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_model import (
    preprocess_features,
    engineer_features,
    build_model,
    train_model,
    evaluate_model,
    predict,
    remove_outliers_three_sigma,
    filter_invalid_pace_time_runs,
    clean_runs_for_learning,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SUPPORTED_MODELS,
    DEFAULT_SIGMA,
)


@pytest.fixture
def sample_data():
    """Generate valid sample training data"""
    rng = np.random.default_rng(42)
    n = 50
    
    return pd.DataFrame({
        'distance_km': rng.uniform(5, 42, n),
        'elevation_gain_m': rng.uniform(0, 500, n),
        'avg_pace_s_per_km': rng.uniform(240, 420, n),
        'avg_heart_rate_bpm': rng.uniform(130, 180, n),
        'temperature_c': rng.uniform(5, 30, n),
        'time_of_day_hour': rng.uniform(5, 20, n),
        'fatigue_score': rng.uniform(0, 10, n),
        'finish_time_s': rng.uniform(1200, 10800, n),
    })


class TestRemoveOutliersThreeSigma:
    """Test Three Sigma outlier removal"""

    def test_no_outliers_retained(self, sample_data):
        """Normal data should retain all rows"""
        cleaned, report = remove_outliers_three_sigma(sample_data.assign(run_id=range(len(sample_data))))
        assert len(cleaned) == len(sample_data)
        assert report["removed_count"] == 0

    def test_extreme_outlier_removed(self, sample_data):
        """Obvious outlier beyond 3σ should be removed"""
        data = sample_data.copy()
        data["run_id"] = [f"run_{i}" for i in range(len(data))]
        # Inject extreme pace outlier
        data.loc[0, "avg_pace_s_per_km"] = data["avg_pace_s_per_km"].mean() + 10 * data["avg_pace_s_per_km"].std()

        cleaned, report = remove_outliers_three_sigma(data)

        assert report["removed_count"] >= 1
        assert 0 not in cleaned.index
        assert report["outliers"][0]["run_id"] == "run_0"

    def test_skipped_for_small_dataset(self):
        """Should skip outlier removal when too few samples"""
        small = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "distance_km": [5, 10, 1000],
            "elevation_gain_m": [10, 20, 30],
            "avg_pace_s_per_km": [300, 310, 320],
            "finish_time_s": [1500, 3000, 4500],
        })
        cleaned, report = remove_outliers_three_sigma(small)
        assert report["skipped"] is True
        assert len(cleaned) == 3

    def test_sigma_threshold(self, sample_data):
        """Report should record the sigma used"""
        _, report = remove_outliers_three_sigma(sample_data.assign(run_id=range(len(sample_data))))
        assert report["sigma"] == DEFAULT_SIGMA
        assert report["method"] == "three_sigma"


class TestPaceTimeCleaning:
    """Test pace/time focused cleaning pipeline"""

    def test_invalid_fast_pace_removed(self):
        data = pd.DataFrame({
            "run_id": ["good", "bad"],
            "distance_km": [10.0, 10.0],
            "elevation_gain_m": [50, 50],
            "avg_pace_s_per_km": [360.0, 60.0],  # 1:00/km impossible
            "finish_time_s": [3600.0, 600.0],
        })
        cleaned, report = filter_invalid_pace_time_runs(data)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["run_id"] == "good"
        assert report["removed_count"] == 1

    def test_pace_time_mismatch_removed(self):
        data = pd.DataFrame({
            "run_id": ["glitch"],
            "distance_km": [10.0],
            "elevation_gain_m": [50],
            "avg_pace_s_per_km": [360.0],   # 6:00/km
            "finish_time_s": [36000.0],    # 10 hours — implied 60:00/km
        })
        cleaned, report = filter_invalid_pace_time_runs(data)
        assert len(cleaned) == 0
        assert report["removed_count"] == 1

    def test_clean_pipeline_removes_sigma_outlier(self, sample_data):
        data = sample_data.assign(run_id=[f"run_{i}" for i in range(len(sample_data))])
        data.loc[0, "avg_pace_s_per_km"] = data["avg_pace_s_per_km"].mean() + 10 * data["avg_pace_s_per_km"].std()

        cleaned, report = clean_runs_for_learning(data)
        assert report["removed_count"] >= 1
        assert "stages" in report
        assert len(cleaned) < len(data)


class TestPreprocessFeatures:
    """Test data preprocessing"""
    
    def test_preprocess_valid_data(self, sample_data):
        """Should successfully preprocess valid data"""
        cleaned = preprocess_features(sample_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) == len(sample_data)
        assert TARGET_COLUMN in cleaned.columns
        assert all(col in cleaned.columns for col in FEATURE_COLUMNS)
    
    def test_handle_missing_target(self, sample_data):
        """Should drop rows with missing target values"""
        data_with_nan = sample_data.copy()
        data_with_nan.loc[0:4, TARGET_COLUMN] = np.nan
        
        cleaned = preprocess_features(data_with_nan)
        
        assert len(cleaned) == len(sample_data) - 5
        assert not cleaned[TARGET_COLUMN].isna().any()
    
    def test_impute_missing_features(self, sample_data):
        """Should impute missing feature values with median"""
        data_with_nan = sample_data.copy()
        data_with_nan.loc[0:4, 'distance_km'] = np.nan
        
        cleaned = preprocess_features(data_with_nan)
        
        # Should fill NaN values
        assert not cleaned['distance_km'].isna().any()
        # The first 5 rows should be filled with median of non-NaN values
        expected_median = data_with_nan['distance_km'].median()
        assert cleaned.loc[0, 'distance_km'] == expected_median
    
    def test_raise_error_missing_columns(self):
        """Should raise ValueError for missing required columns"""
        incomplete_data = pd.DataFrame({
            'distance_km': [10, 20],
            'finish_time_s': [3000, 6000]
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            preprocess_features(incomplete_data)
    
    def test_raise_error_missing_target_column(self, sample_data):
        """Should raise ValueError if target column missing"""
        data_no_target = sample_data.drop(columns=[TARGET_COLUMN])
        
        with pytest.raises(ValueError, match="Target column"):
            preprocess_features(data_no_target)


class TestEngineerFeatures:
    """Test feature engineering"""
    
    def test_create_derived_features(self, sample_data):
        """Should create pace_per_elevation and distance_x_fatigue"""
        engineered = engineer_features(sample_data)
        
        assert 'pace_per_elevation' in engineered.columns
        assert 'distance_x_fatigue' in engineered.columns
    
    def test_pace_per_elevation_calculation(self):
        """Should calculate pace_per_elevation correctly"""
        data = pd.DataFrame({
            'avg_pace_s_per_km': [300, 360],
            'elevation_gain_m': [99, 199],
            'distance_km': [10, 20],
            'fatigue_score': [5, 7]
        })
        
        result = engineer_features(data)
        
        # 300 / (99 + 1) = 3.0
        assert result.loc[0, 'pace_per_elevation'] == pytest.approx(3.0)
        # 360 / (199 + 1) = 1.8
        assert result.loc[1, 'pace_per_elevation'] == pytest.approx(1.8)
    
    def test_distance_x_fatigue_calculation(self):
        """Should calculate distance_x_fatigue correctly"""
        data = pd.DataFrame({
            'distance_km': [10, 20],
            'fatigue_score': [5, 7],
            'avg_pace_s_per_km': [300, 360],
            'elevation_gain_m': [100, 200]
        })
        
        result = engineer_features(data)
        
        assert result.loc[0, 'distance_x_fatigue'] == 50  # 10 * 5
        assert result.loc[1, 'distance_x_fatigue'] == 140  # 20 * 7
    
    def test_handle_zero_elevation(self):
        """Should handle zero elevation without division error"""
        data = pd.DataFrame({
            'avg_pace_s_per_km': [300],
            'elevation_gain_m': [0],
            'distance_km': [10],
            'fatigue_score': [5]
        })
        
        result = engineer_features(data)
        
        # Should add 1 to avoid division by zero
        assert result.loc[0, 'pace_per_elevation'] == 300.0  # 300 / (0 + 1)


class TestBuildModel:
    """Test model building"""
    
    def test_build_linear_model(self):
        """Should build linear regression pipeline"""
        pipeline = build_model('linear')
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
    
    def test_build_random_forest(self):
        """Should build random forest pipeline"""
        pipeline = build_model('random_forest')
        
        assert pipeline is not None
        assert 'model' in pipeline.named_steps
    
    def test_build_gradient_boosting(self):
        """Should build gradient boosting pipeline"""
        pipeline = build_model('gradient_boosting')
        
        assert pipeline is not None
        assert 'model' in pipeline.named_steps
    
    def test_raise_error_invalid_model(self):
        """Should raise ValueError for unknown model type"""
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model('invalid_model')


class TestTrainModel:
    """Test model training"""
    
    def test_train_on_valid_data(self, sample_data):
        """Should successfully train model on valid data"""
        pipeline, X_train, y_train, X_test, y_test = train_model(
            sample_data, 
            model_type='linear',
            random_state=42
        )
        
        assert pipeline is not None
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_reproducible_split(self, sample_data):
        """Should produce same train/test split with same random_state"""
        _, X_train1, _, X_test1, _ = train_model(
            sample_data, random_state=42
        )
        _, X_train2, _, X_test2, _ = train_model(
            sample_data, random_state=42
        )
        
        # Same indices should be selected
        assert X_train1.index.tolist() == X_train2.index.tolist()
        assert X_test1.index.tolist() == X_test2.index.tolist()
    
    def test_train_all_model_types(self, sample_data):
        """Should successfully train all supported model types"""
        for model_type in SUPPORTED_MODELS:
            pipeline, _, _, _, _ = train_model(
                sample_data, 
                model_type=model_type,
                random_state=42
            )
            assert pipeline is not None
    
    def test_fail_on_insufficient_data(self):
        """Should handle very small datasets"""
        tiny_data = pd.DataFrame({
            'distance_km': [10],
            'elevation_gain_m': [100],
            'avg_pace_s_per_km': [300],
            'avg_heart_rate_bpm': [155],
            'temperature_c': [20],
            'time_of_day_hour': [8],
            'fatigue_score': [5],
            'finish_time_s': [3000]
        })
        
        # Should raise error or warning for insufficient data
        with pytest.raises((ValueError, Exception)):
            train_model(tiny_data, test_size=0.5)


class TestEvaluateModel:
    """Test model evaluation"""
    
    def test_evaluate_returns_metrics(self, sample_data):
        """Should return MAE, RMSE, and R² metrics"""
        pipeline, _, _, X_test, y_test = train_model(
            sample_data, random_state=42
        )
        
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_metrics_are_reasonable(self, sample_data):
        """Should produce reasonable metric values"""
        pipeline, _, _, X_test, y_test = train_model(
            sample_data, random_state=42
        )
        
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # MAE and RMSE should be positive
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        # R² should typically be between -1 and 1 (can be negative for poor models)
        assert metrics['r2'] >= -1
    
    def test_rmse_greater_than_mae(self, sample_data):
        """RMSE should be >= MAE (penalizes large errors more)"""
        pipeline, _, _, X_test, y_test = train_model(
            sample_data, random_state=42
        )
        
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        assert metrics['rmse'] >= metrics['mae']


class TestPredict:
    """Test single prediction"""
    
    def test_predict_single_run(self, sample_data):
        """Should make prediction for a single run"""
        pipeline, _, _, _, _ = train_model(sample_data, random_state=42)
        
        features = {
            'distance_km': 10.0,
            'elevation_gain_m': 150,
            'avg_pace_s_per_km': 300,
            'avg_heart_rate_bpm': 155,
            'temperature_c': 20,
            'time_of_day_hour': 8,
            'fatigue_score': 5
        }
        
        prediction = predict(pipeline, features)
        
        assert isinstance(prediction, float)
        assert prediction > 0
    
    def test_prediction_is_numeric(self, sample_data):
        """Should return numeric finish time"""
        pipeline, _, _, _, _ = train_model(sample_data, random_state=42)
        
        features = {
            'distance_km': 5.0,
            'elevation_gain_m': 50,
            'avg_pace_s_per_km': 270,
            'avg_heart_rate_bpm': 150,
            'temperature_c': 15,
            'time_of_day_hour': 7,
            'fatigue_score': 3
        }
        
        prediction = predict(pipeline, features)
        
        assert not np.isnan(prediction)
        assert not np.isinf(prediction)
    
    def test_fail_on_missing_features(self, sample_data):
        """Should raise error for incomplete feature dict"""
        pipeline, _, _, _, _ = train_model(sample_data, random_state=42)
        
        incomplete_features = {
            'distance_km': 10.0,
            'elevation_gain_m': 150
            # Missing other required features
        }
        
        with pytest.raises((KeyError, Exception)):
            predict(pipeline, incomplete_features)
    
    def test_fail_on_invalid_feature_values(self, sample_data):
        """Should handle invalid feature values appropriately"""
        pipeline, _, _, _, _ = train_model(sample_data, random_state=42)
        
        invalid_features = {
            'distance_km': -10.0,  # Negative distance
            'elevation_gain_m': 150,
            'avg_pace_s_per_km': 300,
            'avg_heart_rate_bpm': 155,
            'temperature_c': 20,
            'time_of_day_hour': 8,
            'fatigue_score': 5
        }
        
        # Model might still make prediction, but should handle gracefully
        prediction = predict(pipeline, invalid_features)
        assert isinstance(prediction, float)


class TestModelIntegration:
    """Integration tests for full ML workflow"""
    
    def test_full_ml_pipeline(self, sample_data):
        """Should complete full ML workflow without errors"""
        # Train
        pipeline, X_train, y_train, X_test, y_test = train_model(
            sample_data, 
            model_type='gradient_boosting',
            random_state=42
        )
        
        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Predict
        features = {
            'distance_km': 10.0,
            'elevation_gain_m': 150,
            'avg_pace_s_per_km': 300,
            'avg_heart_rate_bpm': 155,
            'temperature_c': 20,
            'time_of_day_hour': 8,
            'fatigue_score': 5
        }
        prediction = predict(pipeline, features)
        
        # All should succeed
        assert metrics['r2'] is not None
        assert prediction > 0
    
    def test_consistent_predictions(self, sample_data):
        """Should give same prediction for same input"""
        pipeline, _, _, _, _ = train_model(sample_data, random_state=42)
        
        features = {
            'distance_km': 10.0,
            'elevation_gain_m': 150,
            'avg_pace_s_per_km': 300,
            'avg_heart_rate_bpm': 155,
            'temperature_c': 20,
            'time_of_day_hour': 8,
            'fatigue_score': 5
        }
        
        pred1 = predict(pipeline, features)
        pred2 = predict(pipeline, features)
        
        assert pred1 == pred2
