"""
Workload Prediction Model Module
Trains LightGBM Regressor to predict task completion time for employees.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from typing import Dict, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')


def prepare_workload_data(pairs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for workload prediction model.
    
    Args:
        pairs_df: DataFrame with employee-task pairs
    
    Returns:
        Tuple of (X, y)
    """
    print("\nPreparing data for workload prediction model...")
    
    feature_cols = [
        'estimated_hours',
        'difficulty_numeric',
        'complexity_score',
        'experience_years',
        'required_experience',
        'performance_score',
        'success_rate',
        'efficiency_score',
        'current_workload',
        'workload_ratio',
        'availability_hours',
        'skill_similarity_score',
        'department_match',
        'role_alignment',
        'priority_numeric',
        'deadline_days'
    ]
    
    X = pairs_df[feature_cols].copy()
    
    # Create realistic target: predicted completion time
    # Base time = estimated hours, adjusted by employee efficiency and complexity
    base_time = pairs_df['estimated_hours'].values
    
    # Efficiency factor (better performance = faster completion)
    efficiency_factor = 1.0 / (pairs_df['performance_score'] / 10 + 0.5)
    
    # Skill factor (better skill match = faster completion)
    skill_factor = 1.0 / (pairs_df['skill_similarity_score'] + 0.5)
    
    # Workload factor (higher workload = slower completion)
    workload_factor = 1.0 + (pairs_df['workload_ratio'] * 0.3)
    
    # Complexity factor
    complexity_factor = 1.0 + (pairs_df['difficulty_numeric'] - 2) * 0.2
    
    # Calculate predicted time to complete
    predicted_time = (
        base_time * 
        efficiency_factor * 
        skill_factor * 
        workload_factor * 
        complexity_factor
    )
    
    # Add some noise for realism
    np.random.seed(42)
    noise = np.random.normal(0, predicted_time * 0.05)
    y = predicted_time + noise
    y = np.clip(y, base_time * 0.3, base_time * 3.0)  # Keep within reasonable bounds
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}] hours")
    print(f"Mean predicted time: {y.mean():.2f} hours")
    
    return X, y


def train_workload_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    use_optuna: bool = True,
    n_trials: int = 50
) -> Tuple[lgb.LGBMRegressor, Dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train LightGBM regressor for workload prediction with hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector (predicted hours)
        test_size: Test set size
        random_state: Random seed
        use_optuna: Whether to use Optuna for hyperparameter tuning
        n_trials: Number of Optuna trials
    
    Returns:
        Tuple of (model, best_params, X_train, y_train, X_test, y_test)
    """
    print("\n" + "="*60)
    print("TRAINING WORKLOAD PREDICTION MODEL")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    if use_optuna:
        print(f"\nStarting Optuna hyperparameter optimization ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'random_state': random_state,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            model = lgb.LGBMRegressor(**params)
            
            # Cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1
            )
            
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': random_state
        })
        
        print(f"\nBest parameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    
    else:
        # Default parameters
        best_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 8,
            'random_state': random_state,
            'verbosity': -1
        }
    
    # Train final model
    print("\nTraining final model...")
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print("Model training complete!")
    
    return model, best_params, X_train, y_train, X_test, y_test


def evaluate_workload_model(
    model: lgb.LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Evaluate workload prediction model.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mape': train_mape,
        'test_mape': test_mape
    }
    
    # Print metrics
    print("\nRegression Metrics:")
    print(f"  Train MAE:  {train_mae:.4f} hours")
    print(f"  Test MAE:   {test_mae:.4f} hours")
    print(f"  Train RMSE: {train_rmse:.4f} hours")
    print(f"  Test RMSE:  {test_rmse:.4f} hours")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    print(f"  Train MAPE: {train_mape:.2f}%")
    print(f"  Test MAPE:  {test_mape:.2f}%")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.2f}")
    
    # Prediction accuracy within tolerance
    tolerance_1hr = np.mean(np.abs(y_test - y_test_pred) <= 1.0) * 100
    tolerance_2hr = np.mean(np.abs(y_test - y_test_pred) <= 2.0) * 100
    tolerance_3hr = np.mean(np.abs(y_test - y_test_pred) <= 3.0) * 100
    
    print(f"\nPrediction Accuracy:")
    print(f"  Within ±1 hour:  {tolerance_1hr:.2f}%")
    print(f"  Within ±2 hours: {tolerance_2hr:.2f}%")
    print(f"  Within ±3 hours: {tolerance_3hr:.2f}%")
    
    metrics['accuracy_1hr'] = tolerance_1hr
    metrics['accuracy_2hr'] = tolerance_2hr
    metrics['accuracy_3hr'] = tolerance_3hr
    
    return metrics


def save_workload_model(
    model: lgb.LGBMRegressor,
    params: Dict,
    metrics: Dict,
    save_path: str = "../models/workload_model.pkl"
):
    """Save trained workload prediction model and metadata."""
    print(f"\nSaving model to {save_path}...")
    
    model_package = {
        'model': model,
        'params': params,
        'metrics': metrics,
        'feature_names': model.feature_name_
    }
    
    joblib.dump(model_package, save_path)
    print("Model saved successfully!")


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    from feature_engineering import engineer_features_pipeline
    
    # Load and prepare data
    employee_df, task_df = preprocess_pipeline(
        "../data/employee_dataset_532.csv",
        "../data/task_dataset_40.csv"
    )
    
    pairs_df, _, _, _ = engineer_features_pipeline(employee_df, task_df)
    
    # Prepare training data
    X, y = prepare_workload_data(pairs_df)
    
    # Train model
    model, params, X_train, y_train, X_test, y_test = train_workload_model(
        X, y, use_optuna=True, n_trials=30
    )
    
    # Evaluate model
    metrics = evaluate_workload_model(
        model, X_train, y_train, X_test, y_test
    )
    
    # Save model
    save_workload_model(model, params, metrics)
