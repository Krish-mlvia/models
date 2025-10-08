"""
Data Preprocessing Module
Handles loading, cleaning, and initial preprocessing of employee and task datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_datasets(employee_path: str, task_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load employee and task datasets from CSV files.
    
    Args:
        employee_path: Path to employee dataset CSV
        task_path: Path to task dataset CSV
    
    Returns:
        Tuple of (employee_df, task_df)
    """
    print("Loading datasets...")
    employee_df = pd.read_csv(employee_path)
    task_df = pd.read_csv(task_path)
    
    print(f"Employee dataset: {employee_df.shape[0]} rows, {employee_df.shape[1]} columns")
    print(f"Task dataset: {task_df.shape[0]} rows, {task_df.shape[1]} columns")
    
    return employee_df, task_df


def clean_employee_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess employee dataset.
    
    Args:
        df: Raw employee DataFrame
    
    Returns:
        Cleaned employee DataFrame
    """
    print("\nCleaning employee data...")
    df = df.copy()
    
    # Handle missing values
    df['Certifications'] = df['Certifications'].fillna('None')
    df['Skills'] = df['Skills'].fillna('')
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['Salary_INR', 'Experience_Years', 'Performance_1_10', 
                   'Current_Workload_Tasks', 'Availability_Hours_per_Week',
                   'LastProjectSuccessRate', 'suitability_score']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['EmployeeID'])
    
    # Validate data ranges
    df = df[df['Performance_1_10'].between(1, 10)]
    df = df[df['Experience_Years'] >= 0]
    df = df[df['Availability_Hours_per_Week'] > 0]
    
    print(f"Cleaned employee data: {df.shape[0]} rows")
    return df


def clean_task_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess task dataset.
    
    Args:
        df: Raw task DataFrame
    
    Returns:
        Cleaned task DataFrame
    """
    print("\nCleaning task data...")
    df = df.copy()
    
    # Handle missing values
    df['Required_Skills'] = df['Required_Skills'].fillna('')
    df['Description'] = df['Description'].fillna('')
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['Estimated_Hours', 'Deadline_Days', 'Required_Experience']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['TaskID'])
    
    # Map difficulty to numeric values
    difficulty_map = {
        'Easy': 1,
        'Moderate': 2,
        'Hard': 3,
        'Very Hard': 4
    }
    df['Difficulty_Numeric'] = df['Difficulty'].map(difficulty_map)
    
    # Map priority to numeric values
    priority_map = {
        'Low': 1,
        'Medium': 2,
        'High': 3,
        'Critical': 4
    }
    df['Priority_Numeric'] = df['Priority'].map(priority_map)
    
    print(f"Cleaned task data: {df.shape[0]} rows")
    return df


def add_derived_features(employee_df: pd.DataFrame, task_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add derived features to both datasets.
    
    Args:
        employee_df: Cleaned employee DataFrame
        task_df: Cleaned task DataFrame
    
    Returns:
        Tuple of (employee_df, task_df) with derived features
    """
    print("\nAdding derived features...")
    
    # Employee derived features
    employee_df['workload_ratio'] = (
        employee_df['Current_Workload_Tasks'] / 
        (employee_df['Availability_Hours_per_Week'] / 10)
    ).clip(0, 5)
    
    employee_df['efficiency_score'] = (
        employee_df['Performance_1_10'] * 
        employee_df['LastProjectSuccessRate']
    )
    
    employee_df['experience_level'] = pd.cut(
        employee_df['Experience_Years'],
        bins=[-1, 2, 5, 10, 100],
        labels=['Junior', 'Mid', 'Senior', 'Expert']
    )
    
    # Task derived features
    task_df['urgency_score'] = (
        task_df['Priority_Numeric'] / 
        (task_df['Deadline_Days'] + 1)
    )
    
    task_df['complexity_score'] = (
        task_df['Difficulty_Numeric'] * 
        task_df['Estimated_Hours'] / 10
    )
    
    print("Derived features added successfully")
    return employee_df, task_df


def preprocess_pipeline(employee_path: str, task_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline.
    
    Args:
        employee_path: Path to employee dataset CSV
        task_path: Path to task dataset CSV
    
    Returns:
        Tuple of (employee_df, task_df) fully preprocessed
    """
    # Load data
    employee_df, task_df = load_datasets(employee_path, task_path)
    
    # Clean data
    employee_df = clean_employee_data(employee_df)
    task_df = clean_task_data(task_df)
    
    # Add derived features
    employee_df, task_df = add_derived_features(employee_df, task_df)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Final employee dataset: {employee_df.shape}")
    print(f"Final task dataset: {task_df.shape}")
    
    return employee_df, task_df


if __name__ == "__main__":
    # Test preprocessing
    employee_df, task_df = preprocess_pipeline(
        "../data/employee_dataset_532.csv",
        "../data/task_dataset_40.csv"
    )
    
    print("\nEmployee columns:", employee_df.columns.tolist())
    print("\nTask columns:", task_df.columns.tolist())
    print("\nEmployee sample:")
    print(employee_df.head())
    print("\nTask sample:")
    print(task_df.head())
