"""
Model Training Module - Suitability Prediction Model
Trains LightGBM model to predict employee-task suitability scores.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    ndcg_score, precision_score, recall_score, f1_score, roc_auc_score
)
import optuna
from typing import Dict, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')


def prepare_suitability_data(pairs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for suitability model.
    
    Args:
        pairs_df: DataFrame with employee-task pairs
    
    Returns:
        Tuple of (X, y)
    """
    print("\nPreparing data for suitability model...")
    
    feature_cols = [
        'skill_similarity_score',
        'experience_years',
        'required_experience',
        'experience_difference',
        'performance_score',
        'success_rate',
        'current_workload',
        'availability_hours',
        'workload_ratio',
        'efficiency_score',
        'estimated_hours',
        'deadline_days',
        'difficulty_numeric',
        'priority_numeric',
        'urgency_score',
        'complexity_score',
        'department_match',
        'hours_vs_availability',
        'role_alignment'
    ]
    
    X = pairs_df[feature_cols].copy()
    y = pairs_df['suitability_score'].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y


def train_suitability_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    use_optuna: bool = True,
    n_trials: int = 50
) -> Tuple[lgb.LGBMRegressor, Dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train LightGBM model for suitability prediction with hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set size
        random_state: Random seed
        use_optuna: Whether to use Optuna for hyperparameter tuning
        n_trials: Number of Optuna trials
    
    Returns:
        Tuple of (model, best_params, X_train, y_train, X_test, y_test)
    """
    print("\n" + "="*60)
    print("TRAINING SUITABILITY MODEL")
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
                'metric': 'rmse',
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
                cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
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
            'metric': 'rmse',
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
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print("Model training complete!")
    
    return model, best_params, X_train, y_train, X_test, y_test


def evaluate_suitability_model(
    model: lgb.LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pairs_df: pd.DataFrame
) -> Dict:
    """
    Evaluate suitability model with comprehensive metrics.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        pairs_df: Original pairs DataFrame for ranking metrics
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Regression metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    # Classification metrics (for high suitability threshold)
    threshold = 70  # Consider 70+ as suitable match
    y_test_binary = (y_test >= threshold).astype(int)
    y_test_pred_binary = (y_test_pred >= threshold).astype(int)
    
    if len(np.unique(y_test_binary)) > 1:
        precision = precision_score(y_test_binary, y_test_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_test_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_test_pred_binary, zero_division=0)
        
        metrics['precision_at_70'] = precision
        metrics['recall_at_70'] = recall
        metrics['f1_at_70'] = f1
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test_binary, y_test_pred)
            metrics['roc_auc'] = roc_auc
        except:
            metrics['roc_auc'] = None
    
    # Ranking metrics (NDCG)
    # Group by TaskID and compute NDCG@5 and NDCG@10
    test_indices = X_test.index
    test_pairs = pairs_df.loc[test_indices].copy()
    test_pairs['predicted_score'] = y_test_pred
    test_pairs['actual_score'] = y_test.values
    
    ndcg_scores_5 = []
    ndcg_scores_10 = []
    
    for task_id in test_pairs['TaskID'].unique():
        task_group = test_pairs[test_pairs['TaskID'] == task_id]
        
        if len(task_group) >= 5:
            actual = task_group['actual_score'].values.reshape(1, -1)
            predicted = task_group['predicted_score'].values.reshape(1, -1)
            
            try:
                ndcg_5 = ndcg_score(actual, predicted, k=5)
                ndcg_scores_5.append(ndcg_5)
                
                if len(task_group) >= 10:
                    ndcg_10 = ndcg_score(actual, predicted, k=10)
                    ndcg_scores_10.append(ndcg_10)
            except:
                pass
    
    if ndcg_scores_5:
        metrics['ndcg@5'] = np.mean(ndcg_scores_5)
    if ndcg_scores_10:
        metrics['ndcg@10'] = np.mean(ndcg_scores_10)
    
    # Print metrics
    print("\nRegression Metrics:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Train MAE:  {train_mae:.4f}")
    print(f"  Test MAE:   {test_mae:.4f}")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    
    if 'precision_at_70' in metrics:
        print(f"\nClassification Metrics (threshold=70):")
        print(f"  Precision: {metrics['precision_at_70']:.4f}")
        print(f"  Recall:    {metrics['recall_at_70']:.4f}")
        print(f"  F1-Score:  {metrics['f1_at_70']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if 'ndcg@5' in metrics:
        print(f"\nRanking Metrics:")
        print(f"  NDCG@5:  {metrics['ndcg@5']:.4f}")
        if 'ndcg@10' in metrics:
            print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.2f}")
    
    return metrics


def save_suitability_model(
    model: lgb.LGBMRegressor,
    params: Dict,
    metrics: Dict,
    save_path: str = "../models/suitability_model.pkl"
):
    """Save trained suitability model and metadata."""
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
    X, y = prepare_suitability_data(pairs_df)
    
    # Train model
    model, params, X_train, y_train, X_test, y_test = train_suitability_model(
        X, y, use_optuna=True, n_trials=30
    )
    
    # Evaluate model
    metrics = evaluate_suitability_model(
        model, X_train, y_train, X_test, y_test, pairs_df
    )
    
    # Save model
    save_suitability_model(model, params, metrics)
