"""
visualize_model.py

Generates diagnostic visualizations for trained ML models:
- Loss curves (for Gradient Boosting)
- Learning curves
- Feature importance
- Prediction scatter plots

Usage:
    python visualize_model.py
    python visualize_model.py --model models/best_model_gradient_boosting_20260517_173948.pkl
"""

import sys
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
from data_loader import load_all_runs
from ml_model import preprocess_features, engineer_features, FEATURE_COLUMNS, TARGET_COLUMN


def plot_gradient_boosting_loss(model, X_train, y_train, X_val, y_val, output_dir):
    """Plot training loss curve for Gradient Boosting."""
    # Extract the actual estimator from pipeline
    if hasattr(model, 'named_steps'):
        gb_model = model.named_steps['model']
        scaler = model.named_steps['scaler']
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        gb_model = model
        X_train_scaled = X_train
        X_val_scaled = X_val
    
    # Get staged predictions (predictions at each boosting iteration)
    train_scores = []
    val_scores = []
    
    print("   Computing loss at each boosting stage...")
    for i, y_pred in enumerate(gb_model.staged_predict(X_train_scaled)):
        train_scores.append(mean_squared_error(y_train, y_pred))
    
    for i, y_pred in enumerate(gb_model.staged_predict(X_val_scaled)):
        val_scores.append(mean_squared_error(y_val, y_pred))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_scores) + 1), train_scores, label='Training Loss (MSE)', linewidth=2, color='steelblue')
    ax.plot(range(1, len(val_scores) + 1), val_scores, label='Validation Loss (MSE)', linewidth=2, color='tomato')
    ax.axhline(val_scores[-1], color='green', linestyle='--', alpha=0.7, label=f'Final Val MSE: {val_scores[-1]:.0f}')
    ax.set_xlabel('Boosting Iteration (Tree #)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (seconds²)', fontsize=12)
    ax.set_title('Gradient Boosting: Loss Reduction Over Iterations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for convergence
    min_val_idx = np.argmin(val_scores)
    ax.annotate(f'Best iteration: {min_val_idx+1}',
                xy=(min_val_idx+1, val_scores[min_val_idx]),
                xytext=(min_val_idx+1+10, val_scores[min_val_idx]+500),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    loss_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {loss_path}")
    plt.close()
    
    print(f"\n   📉 Training Loss:   Initial={train_scores[0]:.0f} → Final={train_scores[-1]:.0f}  (Reduction: {(1-train_scores[-1]/train_scores[0])*100:.1f}%)")
    print(f"   📉 Validation Loss: Initial={val_scores[0]:.0f} → Final={val_scores[-1]:.0f}  (Reduction: {(1-val_scores[-1]/val_scores[0])*100:.1f}%)")


def plot_feature_importance(model, feature_names, output_dir):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'named_steps'):
        estimator = model.named_steps['model']
    else:
        estimator = model
    
    if not hasattr(estimator, 'feature_importances_'):
        print("   ⚠️  Model doesn't support feature importance")
        return
    
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = ax.barh(range(len(importances)), importances[indices], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance (Which features matter most?)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    importance_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {importance_path}")
    plt.close()
    
    print(f"\n   🏆 TOP 3 MOST IMPORTANT FEATURES:")
    for rank, i in enumerate(indices[:3], 1):
        print(f"      {rank}. {feature_names[i]:<30} (Importance: {importances[i]:.4f})")


def plot_prediction_scatter(model, X, y, dataset_name, output_dir):
    """Scatter plot of predicted vs actual values."""
    y_pred = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate residuals
    residuals = y - y_pred
    mae = np.mean(np.abs(residuals))
    
    # Color points by error magnitude
    colors = np.abs(residuals)
    scatter = ax.scatter(y, y_pred, c=colors, cmap='RdYlGn_r', alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction', zorder=5)
    
    ax.set_xlabel('Actual Finish Time (seconds)', fontsize=12)
    ax.set_ylabel('Predicted Finish Time (seconds)', fontsize=12)
    ax.set_title(f'{dataset_name} Set: Predictions vs Reality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Error (seconds)', fontsize=10)
    
    # Add MAE text
    ax.text(0.05, 0.95, f'MAE: {mae:.0f}s', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    scatter_path = os.path.join(output_dir, f'scatter_{dataset_name.lower()}.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {scatter_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize ML model diagnostics")
    parser.add_argument('--model', type=str, help='Path to saved model .pkl file')
    parser.add_argument('--gpx-dir', type=str, default='data/raw_gpx', help='GPX directory')
    parser.add_argument('--output', '-o', type=str, default='models', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"ML MODEL DIAGNOSTICS & VISUALIZATION")
    print(f"{'='*80}\n")
    
    # Load data
    print("📂 Loading GPX data...")
    all_runs_df = load_all_runs(args.gpx_dir, smoothing_window=3)
    
    # Split data (same as training)
    train_df, temp_df = train_test_split(all_runs_df, test_size=0.2, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)
    
    # Prepare features
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
    
    # Load model
    if args.model:
        model_path = args.model
    else:
        # Find most recent model
        model_files = [f for f in os.listdir(args.output) if f.startswith('best_model_') and f.endswith('.pkl')]
        if not model_files:
            print("❌ No model found. Train a model first with: python train_ml_model.py")
            return
        model_path = os.path.join(args.output, sorted(model_files)[-1])
    
    print(f"📦 Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"\n{'='*80}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # 1. Loss curve (for Gradient Boosting)
    if 'gradient_boosting' in model_path.lower():
        print("📊 1. Gradient Boosting Loss Curve")
        plot_gradient_boosting_loss(model, X_train, y_train, X_val, y_val, args.output)
    
    # 2. Feature importance
    print("\n📊 2. Feature Importance")
    plot_feature_importance(model, feature_cols, args.output)
    
    # 3. Prediction scatter plots
    print("\n📊 3. Validation Set Predictions")
    plot_prediction_scatter(model, X_val, y_val, 'Validation', args.output)
    
    print("\n📊 4. Test Set Predictions")
    plot_prediction_scatter(model, X_test, y_test, 'Test', args.output)
    
    print(f"\n{'='*80}")
    print(f"✅ ALL VISUALIZATIONS SAVED TO: {args.output}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
