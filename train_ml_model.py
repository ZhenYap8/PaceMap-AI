"""
train_ml_model.py

Trains ML models on all GPX runs to predict finish times.

Features:
- Loads all runs from data/raw_gpx/
- 80/10/10 train/validation/test split
- Trains Linear Regression, Random Forest, Gradient Boosting
- Saves best model to disk
- Provides prediction function

Usage:
    python train_ml_model.py
    python train_ml_model.py --gpx-dir data/raw_gpx --output models
"""

import sys
import os
import argparse
import pickle
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sklearn.model_selection import train_test_split
from data_loader import load_all_runs
from ml_model import (
    evaluate_model, predict, TARGET_COLUMN, SUPPORTED_MODELS,
    preprocess_features, engineer_features, FEATURE_COLUMNS, build_model,
    clean_runs_for_learning,
)
from utils import format_duration


def train_models(gpx_dir: str, output_dir: str = "models", smoothing_window: int = 3):
    """
    Train ML models on all GPX files and save the best model.
    
    Args:
        gpx_dir: Directory containing GPX files
        output_dir: Directory to save trained models
        smoothing_window: GPS smoothing window
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MACHINE LEARNING MODEL TRAINING")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # STEP 1: Load All GPX Files
    # =========================================================================
    print(f"📂 Loading all GPX files from: {gpx_dir}")
    all_runs_df = load_all_runs(gpx_dir, smoothing_window=smoothing_window)
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully loaded {len(all_runs_df)} runs")
    print(f"{'='*80}\n")

    all_runs_df, outlier_report = clean_runs_for_learning(all_runs_df, id_column="run_id")
    if outlier_report["removed_count"]:
        print(f"🧹 Pace/time cleaning (3σ rule, σ={outlier_report['sigma']}):")
        print(f"   Ignored {outlier_report['removed_count']} of {outlier_report['total_before']} runs "
              f"({outlier_report['removed_pct']}%)")
        for stage in outlier_report.get("stages", []):
            if stage.get("removed_count"):
                print(f"   [{stage['stage']}] {stage.get('description', '')}")
                for o in stage.get("ignored_runs", []):
                    detail = f"{o.get('avg_pace', '')} / {o.get('finish_time', '')}".strip(" /")
                    print(f"     - {o.get('run_id', '?')} ({detail}): {', '.join(o['reasons'])}")
        print()
    elif outlier_report.get("skipped"):
        print(f"ℹ️  Pace/time cleaning skipped: {outlier_report.get('skip_reason')}\n")
    
    print("Dataset preview:")
    print(all_runs_df[['run_id', 'distance_km', 'elevation_gain_m', 'finish_time_s']].head(10))
    
    print("\nDataset statistics:")
    print(all_runs_df[['distance_km', 'elevation_gain_m', 'avg_pace_s_per_km', 'finish_time_s']].describe())
    
    # =========================================================================
    # STEP 2: Split Data (80/10/10)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"SPLITTING DATA (80% Train / 10% Validation / 10% Test)")
    print(f"{'='*80}")
    
    train_df, temp_df = train_test_split(
        all_runs_df, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        shuffle=True
    )
    
    print(f"  Training set   : {len(train_df):2d} runs ({len(train_df)/len(all_runs_df)*100:.0f}%)")
    print(f"  Validation set : {len(val_df):2d} runs ({len(val_df)/len(all_runs_df)*100:.0f}%)")
    print(f"  Test set       : {len(test_df):2d} runs ({len(test_df)/len(all_runs_df)*100:.0f}%)")
    print(f"\nTest runs (held out for final evaluation):")
    print(f"  {', '.join(sorted(test_df['run_id'].tolist()))}")
    
    # =========================================================================
    # STEP 3: Prepare Data
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"PREPARING DATA")
    print(f"{'='*80}")
    
    train_clean = preprocess_features(train_df.drop(columns=['run_id']))
    train_clean = engineer_features(train_clean)
    feature_cols = FEATURE_COLUMNS + ['pace_per_elevation', 'distance_x_fatigue']
    X_train = train_clean[feature_cols]
    y_train = train_clean[TARGET_COLUMN]
    
    val_clean = preprocess_features(val_df.drop(columns=['run_id']))
    val_clean = engineer_features(val_clean)
    X_val = val_clean[feature_cols]
    y_val = val_clean[TARGET_COLUMN]
    
    test_clean = preprocess_features(test_df.drop(columns=['run_id']))
    test_clean = engineer_features(test_clean)
    X_test = test_clean[feature_cols]
    y_test = test_clean[TARGET_COLUMN]
    
    print(f"  Features: {len(feature_cols)} total")
    print(f"    Original: {', '.join(FEATURE_COLUMNS)}")
    print(f"    Engineered: pace_per_elevation, distance_x_fatigue")
    
    # =========================================================================
    # STEP 4: Train All Models
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"TRAINING {len(SUPPORTED_MODELS)} MODELS ON {len(X_train)} RUNS")
    print(f"{'='*80}\n")
    
    results = {}
    for model_type in SUPPORTED_MODELS:
        print(f"Training {model_type}...")
        pipeline = build_model(model_type)
        pipeline.fit(X_train, y_train)
        
        val_metrics = evaluate_model(pipeline, X_val, y_val)
        results[model_type] = {
            'pipeline': pipeline,
            'metrics': val_metrics
        }
        
        print(f"  {model_type:<25} Validation → MAE={val_metrics['mae']:6.0f}s  RMSE={val_metrics['rmse']:6.0f}s  R²={val_metrics['r2']:.4f}")
    
    # =========================================================================
    # STEP 5: Select Best Model
    # =========================================================================
    best_type = max(results, key=lambda m: results[m]['metrics']['r2'])
    best_pipeline = results[best_type]['pipeline']
    best_val_metrics = results[best_type]['metrics']
    
    print(f"\n{'='*80}")
    print(f"✓ BEST MODEL: {best_type.upper()}")
    print(f"{'='*80}")
    print(f"  Validation R²  : {best_val_metrics['r2']:.4f}")
    print(f"  Validation MAE : {best_val_metrics['mae']:.0f}s ({format_duration(best_val_metrics['mae'])})")
    print(f"  Validation RMSE: {best_val_metrics['rmse']:.0f}s ({format_duration(best_val_metrics['rmse'])})")
    
    # =========================================================================
    # STEP 6: Final Test Set Evaluation
    # =========================================================================
    test_metrics = evaluate_model(best_pipeline, X_test, y_test)
    
    print(f"\n{'='*80}")
    print(f"FINAL TEST SET PERFORMANCE ({len(test_df)} unseen runs)")
    print(f"{'='*80}")
    print(f"  MAE  : {test_metrics['mae']:6.0f}s  ({format_duration(test_metrics['mae'])})")
    print(f"  RMSE : {test_metrics['rmse']:6.0f}s  ({format_duration(test_metrics['rmse'])})")
    print(f"  R²   : {test_metrics['r2']:.4f}")
    
    # Show individual predictions
    print(f"\n{'='*80}")
    print(f"TEST SET PREDICTIONS vs ACTUAL")
    print(f"{'='*80}")
    print(f"{'Run ID':<15} {'Predicted':<15} {'Actual':<15} {'Error':<15} {'% Error'}")
    print(f"{'-'*80}")
    
    for idx in test_df.index:
        row = test_df.loc[idx]
        feat = row.drop(['finish_time_s', 'run_id']).to_dict()
        pred = predict(best_pipeline, feat)
        actual = row['finish_time_s']
        error = abs(pred - actual)
        error_pct = (error / actual) * 100
        
        print(f"{row['run_id']:<15} {format_duration(pred):<15} {format_duration(actual):<15} {format_duration(error):<15} {error_pct:>6.1f}%")
    
    print(f"{'-'*80}")
    
    # =========================================================================
    # STEP 7: Save Model
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"best_model_{best_type}_{timestamp}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    
    # Save metadata
    metadata = {
        'model_type': best_type,
        'trained_on': len(train_df),
        'val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'feature_columns': feature_cols,
        'timestamp': timestamp,
    }
    
    metadata_path = os.path.join(output_dir, f"model_metadata_{timestamp}.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n{'='*80}")
    print(f"✅ MODEL SAVED")
    print(f"{'='*80}")
    print(f"  Model file : {model_path}")
    print(f"  Metadata   : {metadata_path}")
    print(f"{'='*80}\n")
    
    return best_pipeline, metadata


def predict_new_run(model_path: str, distance_km: float, elevation_gain_m: float, avg_pace_s_per_km: float):
    """
    Load a trained model and make a prediction for a new run.
    
    Args:
        model_path: Path to saved model file (.pkl)
        distance_km: Planned distance in km
        elevation_gain_m: Expected elevation gain in metres
        avg_pace_s_per_km: Target pace in seconds per km
    """
    # Load model
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    # Create feature dict
    features = {
        'distance_km': distance_km,
        'elevation_gain_m': elevation_gain_m,
        'avg_pace_s_per_km': avg_pace_s_per_km,
        'avg_heart_rate_bpm': 155.0,  # Placeholder
        'temperature_c': 20.0,         # Placeholder
        'time_of_day_hour': 8,         # Placeholder
        'fatigue_score': 5.0,          # Placeholder
    }
    
    # Make prediction
    predicted_time = predict(pipeline, features)
    
    print(f"\n{'='*80}")
    print(f"PREDICTION FOR NEW RUN")
    print(f"{'='*80}")
    print(f"  Distance       : {distance_km:.2f} km")
    print(f"  Elevation Gain : {elevation_gain_m:.1f} m")
    print(f"  Target Pace    : {int(avg_pace_s_per_km//60)}:{int(avg_pace_s_per_km%60):02d} min/km")
    print(f"{'='*80}")
    print(f"  Predicted Time : {format_duration(predicted_time)}")
    print(f"{'='*80}\n")
    
    return predicted_time


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models on GPX data to predict finish times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on all runs
  python train_ml_model.py
  
  # Specify custom directory
  python train_ml_model.py --gpx-dir data/raw_gpx --output models
  
  # Make prediction with saved model
  python train_ml_model.py --predict --model models/best_model_gradient_boosting_20260517_143022.pkl --distance 10 --elevation 100 --pace 300
        """
    )
    
    parser.add_argument(
        '--gpx-dir',
        type=str,
        default='data/raw_gpx',
        help='Directory containing GPX files (default: data/raw_gpx)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for trained models (default: models)'
    )
    
    parser.add_argument(
        '--smoothing', '-s',
        type=int,
        default=3,
        help='GPS smoothing window size (default: 3)'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make prediction with a saved model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to saved model file for prediction'
    )
    
    parser.add_argument(
        '--distance',
        type=float,
        help='Distance in km for prediction'
    )
    
    parser.add_argument(
        '--elevation',
        type=float,
        help='Elevation gain in metres for prediction'
    )
    
    parser.add_argument(
        '--pace',
        type=float,
        help='Target pace in seconds per km for prediction'
    )
    
    args = parser.parse_args()
    
    if args.predict:
        # Prediction mode
        if not all([args.model, args.distance, args.elevation, args.pace]):
            print("❌ Error: For prediction, you must provide --model, --distance, --elevation, and --pace")
            sys.exit(1)
        
        if not os.path.exists(args.model):
            print(f"❌ Error: Model file not found: {args.model}")
            sys.exit(1)
        
        predict_new_run(args.model, args.distance, args.elevation, args.pace)
    else:
        # Training mode
        if not os.path.exists(args.gpx_dir):
            print(f"❌ Error: GPX directory not found: {args.gpx_dir}")
            sys.exit(1)
        
        train_models(args.gpx_dir, args.output, args.smoothing)


if __name__ == "__main__":
    main()
