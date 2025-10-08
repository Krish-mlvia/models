"""
Visualization Module
Generate plots and SHAP analysis for model interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    save_path: str = None
) -> plt.Figure:
    """
    Plot feature importance from LightGBM model.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=feature_importance,
        y='feature',
        x='importance',
        palette='viridis',
        ax=ax
    )
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_shap_values(
    model,
    X: pd.DataFrame,
    max_display: int = 15,
    save_path: str = None
):
    """
    Generate SHAP summary plot for model interpretation.
    
    Args:
        model: Trained model
        X: Feature matrix
        max_display: Maximum features to display
        save_path: Path to save plot
    """
    print("\nGenerating SHAP values...")
    print("This may take a few minutes...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample (full dataset can be slow)
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Feature Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP plot saved to {save_path}")
    
    plt.show()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction Distribution",
    save_path: str = None
) -> plt.Figure:
    """
    Plot distribution of predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution plot saved to {save_path}")
    
    return fig


def plot_assignment_statistics(
    assignments: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot comprehensive assignment statistics.
    
    Args:
        assignments: DataFrame with final assignments
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Task Assignment Statistics', fontsize=18, fontweight='bold')
    
    # 1. Suitability Score Distribution
    axes[0, 0].hist(assignments['SuitabilityScore'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(assignments['SuitabilityScore'].mean(), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f"Mean: {assignments['SuitabilityScore'].mean():.2f}")
    axes[0, 0].set_xlabel('Suitability Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Suitability Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predicted Hours Distribution
    axes[0, 1].hist(assignments['PredictedHours'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(assignments['PredictedHours'].mean(), 
                       color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {assignments['PredictedHours'].mean():.2f}")
    axes[0, 1].set_xlabel('Predicted Hours')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Predicted Hours Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Priority Distribution
    priority_counts = assignments['Priority'].value_counts()
    priority_order = ['Critical', 'High', 'Medium', 'Low']
    priority_counts = priority_counts.reindex(priority_order, fill_value=0)
    
    colors = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#2ca02c', 'Low': '#1f77b4'}
    color_list = [colors[p] for p in priority_order]
    
    axes[0, 2].bar(priority_order, priority_counts.values, color=color_list, edgecolor='black')
    axes[0, 2].set_xlabel('Priority Level')
    axes[0, 2].set_ylabel('Number of Tasks')
    axes[0, 2].set_title('Tasks by Priority')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Difficulty Distribution
    diff_counts = assignments['Difficulty'].value_counts()
    axes[1, 0].bar(diff_counts.index, diff_counts.values, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Difficulty Level')
    axes[1, 0].set_ylabel('Number of Tasks')
    axes[1, 0].set_title('Tasks by Difficulty')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Workload Distribution (tasks per employee)
    if 'AssignedEmployeeID' in assignments.columns:
        tasks_per_emp = assignments.groupby('AssignedEmployeeID').size()
        axes[1, 1].hist(tasks_per_emp.values, bins=max(tasks_per_emp), 
                       color='plum', edgecolor='black', align='left')
        axes[1, 1].set_xlabel('Tasks per Employee')
        axes[1, 1].set_ylabel('Number of Employees')
        axes[1, 1].set_title('Workload Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Suitability vs Predicted Hours
    axes[1, 2].scatter(assignments['SuitabilityScore'], 
                      assignments['PredictedHours'],
                      c=assignments['Priority'].map({'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}),
                      cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1, 2].set_xlabel('Suitability Score')
    axes[1, 2].set_ylabel('Predicted Hours')
    axes[1, 2].set_title('Suitability vs Time')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Assignment statistics plot saved to {save_path}")
    
    return fig


def generate_all_visualizations(
    suitability_model_path: str = "models/suitability_model.pkl",
    workload_model_path: str = "models/workload_model.pkl",
    assignments_path: str = "outputs/final_assignments.csv",
    output_dir: str = "outputs"
):
    """
    Generate all visualization plots.
    
    Args:
        suitability_model_path: Path to suitability model
        workload_model_path: Path to workload model
        assignments_path: Path to assignments CSV
        output_dir: Directory to save plots
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    suit_package = joblib.load(suitability_model_path)
    work_package = joblib.load(workload_model_path)
    
    suit_model = suit_package['model']
    work_model = work_package['model']
    
    # Load assignments
    assignments = pd.read_csv(assignments_path)
    
    # Generate plots
    print("\nGenerating feature importance plots...")
    plot_feature_importance(
        suit_model,
        suit_model.feature_name_,
        top_n=15,
        save_path=f"{output_dir}/suitability_feature_importance.png"
    )
    plt.close()
    
    plot_feature_importance(
        work_model,
        work_model.feature_name_,
        top_n=15,
        save_path=f"{output_dir}/workload_feature_importance.png"
    )
    plt.close()
    
    print("\nGenerating assignment statistics plot...")
    plot_assignment_statistics(
        assignments,
        save_path=f"{output_dir}/assignment_statistics.png"
    )
    plt.close()
    
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    generate_all_visualizations()
