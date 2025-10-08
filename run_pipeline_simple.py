"""
Simple Pipeline Runner
Executes the complete ML pipeline step by step.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("="*80)
print(" WORKFORCE OPTIMIZATION ML PIPELINE")
print("="*80)
print()

# Step 1: Data Preprocessing
print("Step 1/5: Data Preprocessing")
print("-" * 80)
from data_preprocessing import preprocess_pipeline

employee_df, task_df = preprocess_pipeline(
    os.path.join(DATA_DIR, "employee_dataset_532.csv"),
    os.path.join(DATA_DIR, "task_dataset_40.csv")
)
print("✓ Data preprocessing complete\n")

# Step 2: Feature Engineering
print("Step 2/5: Feature Engineering (this may take 2-3 minutes)")
print("-" * 80)
from feature_engineering import engineer_features_pipeline

pairs_df, emp_embeddings, task_embeddings, sim_matrix = engineer_features_pipeline(
    employee_df, task_df
)
print("✓ Feature engineering complete\n")

# Step 3: Train Suitability Model
print("Step 3/5: Training Suitability Model (5-10 minutes)")
print("-" * 80)
from train_suitability_model import prepare_suitability_data, train_suitability_model

X, y = prepare_suitability_data(pairs_df)
suitability_model, best_params, X_train, y_train, X_test, y_test = train_suitability_model(
    X, y, use_optuna=True, n_trials=30
)

# Save suitability model
import joblib
suitability_model_path = os.path.join(MODELS_DIR, 'suitability_model.pkl')
joblib.dump({
    'model': suitability_model,
    'best_params': best_params,
    'feature_names': X.columns.tolist()
}, suitability_model_path)
print(f"✓ Suitability model trained and saved to {suitability_model_path}\n")

# Step 4: Train Workload Model
print("Step 4/5: Training Workload Model (3-5 minutes)")
print("-" * 80)
from train_workload_model import prepare_workload_data, train_workload_model

# Create simulated actual completion times for workload model
import numpy as np
np.random.seed(42)
pairs_df['actual_completion_hours'] = (
    pairs_df['estimated_hours'] * 
    (1 + np.random.normal(0, 0.15, len(pairs_df))) *  # Add realistic variation
    (1.2 - 0.01 * pairs_df['performance_score']) *  # Higher performance = faster
    (1 + 0.05 * pairs_df['workload_ratio'])  # Higher workload = slower
).clip(1, None)

X_work, y_work = prepare_workload_data(pairs_df)
workload_model, work_params, X_work_train, y_work_train, X_work_test, y_work_test = train_workload_model(
    X_work, y_work, use_optuna=True, n_trials=30
)

# Save workload model
workload_model_path = os.path.join(MODELS_DIR, 'workload_model.pkl')
joblib.dump({
    'model': workload_model,
    'best_params': work_params,
    'feature_names': X_work.columns.tolist()
}, workload_model_path)
print(f"✓ Workload model trained and saved to {workload_model_path}\n")

# Step 5: Generate Assignment Assignments
print("Step 5/5: Optimizing Task Assignments")
print("-" * 80)
from assignment_optimizer import optimize_assignments

# Get all predictions
pairs_df['predicted_suitability'] = suitability_model.predict(X)
pairs_df['predicted_hours'] = workload_model.predict(X_work)

# Optimize assignments
assignments_df = optimize_assignments(
    pairs_df, 
    employee_df, 
    task_df,
    method='ilp',
    max_tasks_per_employee=2
)

# Save assignments
assignments_path = os.path.join(OUTPUTS_DIR, 'final_assignments.csv')
assignments_df.to_csv(assignments_path, index=False)
print(f"✓ Assignments saved to {assignments_path}\n")

# Step 6: Generate Metrics Report
print("Generating metrics report...")
print("-" * 80)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from datetime import datetime

# Evaluate models
suit_test_pred = suitability_model.predict(X_test)
work_test_pred = workload_model.predict(X_work_test)

metrics_report = f"""
================================================================================
                    WORKFORCE OPTIMIZATION ML MODELS
                         PERFORMANCE METRICS REPORT
================================================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------------------------------------
                         SUITABILITY MODEL METRICS
--------------------------------------------------------------------------------
Model Type: LightGBM Regressor
Training Samples: {len(X_train)}
Test Samples: {len(X_test)}

Test Set Performance:
  - R² Score:  {r2_score(y_test, suit_test_pred):.4f}
  - RMSE:      {np.sqrt(mean_squared_error(y_test, suit_test_pred)):.4f}
  - MAE:       {mean_absolute_error(y_test, suit_test_pred):.4f}

Train Set Performance:
  - R² Score:  {r2_score(y_train, suitability_model.predict(X_train)):.4f}
  - RMSE:      {np.sqrt(mean_squared_error(y_train, suitability_model.predict(X_train))):.4f}
  - MAE:       {mean_absolute_error(y_train, suitability_model.predict(X_train)):.4f}

Top 5 Important Features:
"""

# Add feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': suitability_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    metrics_report += f"  {idx+1}. {row['feature']}: {row['importance']:.2f}\n"

metrics_report += f"""

--------------------------------------------------------------------------------
                         WORKLOAD MODEL METRICS
--------------------------------------------------------------------------------
Model Type: LightGBM Regressor
Training Samples: {len(X_work_train)}
Test Samples: {len(X_work_test)}

Test Set Performance:
  - R² Score:  {r2_score(y_work_test, work_test_pred):.4f}
  - RMSE:      {np.sqrt(mean_squared_error(y_work_test, work_test_pred)):.4f} hours
  - MAE:       {mean_absolute_error(y_work_test, work_test_pred):.4f} hours

Train Set Performance:
  - R² Score:  {r2_score(y_work_train, workload_model.predict(X_work_train)):.4f}
  - RMSE:      {np.sqrt(mean_squared_error(y_work_train, workload_model.predict(X_work_train))):.4f} hours
  - MAE:       {mean_absolute_error(y_work_train, workload_model.predict(X_work_train)):.4f} hours

--------------------------------------------------------------------------------
                            ASSIGNMENT STATISTICS
--------------------------------------------------------------------------------
Total Tasks: {len(task_df)}
Tasks Assigned: {len(assignments_df)}
Unique Employees Assigned: {assignments_df['AssignedEmployeeID'].nunique()}
Average Suitability Score: {assignments_df['SuitabilityScore'].mean():.2f}
Average Predicted Hours: {assignments_df['PredictedHours'].mean():.2f}

Assignments by Priority:
"""

for priority in ['Critical', 'High', 'Medium', 'Low']:
    count = len(assignments_df[assignments_df['Priority'] == priority])
    metrics_report += f"  - {priority}: {count} tasks\n"

metrics_report += f"""

Assignments by Department:
"""

for dept in assignments_df['Department'].value_counts().head(5).items():
    metrics_report += f"  - {dept[0]}: {dept[1]} tasks\n"

metrics_report += """

================================================================================
                              END OF REPORT
================================================================================
"""

# Save metrics report
metrics_report_path = os.path.join(OUTPUTS_DIR, 'metrics_report.txt')
with open(metrics_report_path, 'w') as f:
    f.write(metrics_report)

print(metrics_report)

print("\n" + "="*80)
print(" ✅ PIPELINE COMPLETE!")
print("="*80)
print("\n📁 Outputs saved to:")
print("  - models/suitability_model.pkl")
print("  - models/workload_model.pkl")
print("  - outputs/final_assignments.csv")
print("  - outputs/metrics_report.txt")
print("\n🎉 All models trained successfully!")
print("="*80)
