"""
Task Assignment Optimization Module
Uses optimization algorithms to assign tasks to employees optimally.
"""

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from ortools.linear_solver import pywraplp
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_cost_matrix(
    employee_df: pd.DataFrame,
    task_df: pd.DataFrame,
    suitability_scores: np.ndarray,
    predicted_hours: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.4
) -> np.ndarray:
    """
    Create cost matrix for assignment optimization.
    
    Args:
        employee_df: Employee DataFrame
        task_df: Task DataFrame
        suitability_scores: Matrix of suitability scores (n_employees x n_tasks)
        predicted_hours: Matrix of predicted completion hours (n_employees x n_tasks)
        alpha: Weight for suitability score (higher = prioritize skill match)
        beta: Weight for time efficiency (higher = prioritize quick completion)
    
    Returns:
        Cost matrix (n_employees x n_tasks) - lower is better
    """
    print("\nCreating cost matrix for optimization...")
    
    n_employees = len(employee_df)
    n_tasks = len(task_df)
    
    # Normalize suitability scores to [0, 1]
    norm_suitability = suitability_scores / 100.0
    
    # Normalize predicted hours relative to availability
    availability_matrix = np.tile(
        employee_df['Availability_Hours_per_Week'].values.reshape(-1, 1),
        (1, n_tasks)
    )
    norm_hours = predicted_hours / (availability_matrix + 1e-6)
    norm_hours = np.clip(norm_hours, 0, 2)  # Clip to reasonable range
    
    # Cost = maximize suitability + minimize time
    # We use negative because assignment algorithm minimizes cost
    cost_matrix = -alpha * norm_suitability + beta * norm_hours
    
    # Add penalties for infeasible assignments
    for i, emp in employee_df.iterrows():
        for j, task in task_df.iterrows():
            # Penalty for department mismatch
            if emp['Department'] != task['Department']:
                cost_matrix[i, j] += 0.2
            
            # Penalty for insufficient experience
            if emp['Experience_Years'] < task['Required_Experience'] - 2:
                cost_matrix[i, j] += 0.3
            
            # Penalty for overload
            if emp['Current_Workload_Tasks'] >= 8:
                cost_matrix[i, j] += 0.4
            
            # Penalty if predicted hours exceed availability
            if predicted_hours[i, j] > emp['Availability_Hours_per_Week']:
                cost_matrix[i, j] += 0.5
    
    print(f"Cost matrix shape: {cost_matrix.shape}")
    print(f"Cost range: [{cost_matrix.min():.4f}, {cost_matrix.max():.4f}]")
    
    return cost_matrix


def hungarian_assignment(
    cost_matrix: np.ndarray,
    employee_df: pd.DataFrame,
    task_df: pd.DataFrame,
    suitability_scores: np.ndarray,
    predicted_hours: np.ndarray
) -> pd.DataFrame:
    """
    Solve one-to-one task assignment using Hungarian Algorithm.
    
    Args:
        cost_matrix: Cost matrix for assignment
        employee_df: Employee DataFrame
        task_df: Task DataFrame
        suitability_scores: Suitability score matrix
        predicted_hours: Predicted hours matrix
    
    Returns:
        DataFrame with optimal assignments
    """
    print("\n" + "="*60)
    print("HUNGARIAN ALGORITHM OPTIMIZATION")
    print("="*60)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create assignments DataFrame
    assignments = []
    
    for emp_idx, task_idx in zip(row_ind, col_ind):
        emp = employee_df.iloc[emp_idx]
        task = task_df.iloc[task_idx]
        
        assignment = {
            'TaskID': task['TaskID'],
            'TaskName': task['TaskName'],
            'AssignedEmployeeID': emp['EmployeeID'],
            'EmployeeName': emp['Name'],
            'EmployeeRole': emp['Role'],
            'Department': task['Department'],
            'SuitabilityScore': suitability_scores[emp_idx, task_idx],
            'PredictedHours': predicted_hours[emp_idx, task_idx],
            'EstimatedHours': task['Estimated_Hours'],
            'DeadlineDays': task['Deadline_Days'],
            'Priority': task['Priority'],
            'Difficulty': task['Difficulty'],
            'EmployeeExperience': emp['Experience_Years'],
            'RequiredExperience': task['Required_Experience'],
            'EmployeePerformance': emp['Performance_1_10'],
            'EmployeeWorkload': emp['Current_Workload_Tasks'],
            'EmployeeAvailability': emp['Availability_Hours_per_Week'],
            'AssignmentCost': cost_matrix[emp_idx, task_idx]
        }
        
        assignments.append(assignment)
    
    assignments_df = pd.DataFrame(assignments)
    
    # Sort by priority and deadline
    priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    assignments_df['PriorityOrder'] = assignments_df['Priority'].map(priority_order)
    assignments_df = assignments_df.sort_values(['PriorityOrder', 'DeadlineDays'])
    assignments_df = assignments_df.drop('PriorityOrder', axis=1)
    
    # Print summary
    print(f"\nTotal assignments: {len(assignments_df)}")
    print(f"Average suitability score: {assignments_df['SuitabilityScore'].mean():.2f}")
    print(f"Average predicted hours: {assignments_df['PredictedHours'].mean():.2f}")
    print(f"\nAssignments by priority:")
    print(assignments_df['Priority'].value_counts().sort_index())
    
    return assignments_df


def ilp_assignment_with_constraints(
    employee_df: pd.DataFrame,
    task_df: pd.DataFrame,
    suitability_scores: np.ndarray,
    predicted_hours: np.ndarray,
    max_tasks_per_employee: int = 3,
    alpha: float = 0.6,
    beta: float = 0.4
) -> pd.DataFrame:
    """
    Solve task assignment with multiple constraints using Integer Linear Programming.
    Allows multiple tasks per employee with workload and fairness constraints.
    
    Args:
        employee_df: Employee DataFrame
        task_df: Task DataFrame
        suitability_scores: Suitability score matrix
        predicted_hours: Predicted hours matrix
        max_tasks_per_employee: Maximum tasks per employee
        alpha: Weight for suitability
        beta: Weight for time efficiency
    
    Returns:
        DataFrame with optimal assignments
    """
    print("\n" + "="*60)
    print("INTEGER LINEAR PROGRAMMING OPTIMIZATION")
    print("="*60)
    print(f"Max tasks per employee: {max_tasks_per_employee}")
    
    n_employees = len(employee_df)
    n_tasks = len(task_df)
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("ERROR: Could not create solver")
        return pd.DataFrame()
    
    # Decision variables: x[i][j] = 1 if employee i is assigned to task j
    x = {}
    for i in range(n_employees):
        for j in range(n_tasks):
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')
    
    # Objective: maximize suitability - minimize time
    objective = solver.Objective()
    
    for i in range(n_employees):
        for j in range(n_tasks):
            emp = employee_df.iloc[i]
            task = task_df.iloc[j]
            
            # Compute score
            suitability = suitability_scores[i, j] / 100.0
            time_factor = predicted_hours[i, j] / (emp['Availability_Hours_per_Week'] + 1e-6)
            
            score = alpha * suitability - beta * time_factor
            
            # Add penalties
            if emp['Department'] != task['Department']:
                score -= 0.2
            if emp['Experience_Years'] < task['Required_Experience'] - 2:
                score -= 0.3
            
            objective.SetCoefficient(x[i, j], score)
    
    objective.SetMaximization()
    
    # Constraint 1: Each task must be assigned to exactly one employee
    for j in range(n_tasks):
        solver.Add(sum(x[i, j] for i in range(n_employees)) == 1)
    
    # Constraint 2: Each employee can be assigned at most max_tasks
    for i in range(n_employees):
        solver.Add(sum(x[i, j] for j in range(n_tasks)) <= max_tasks_per_employee)
    
    # Constraint 3: Respect employee availability (hours)
    for i in range(n_employees):
        emp = employee_df.iloc[i]
        total_hours = sum(
            x[i, j] * predicted_hours[i, j] 
            for j in range(n_tasks)
        )
        solver.Add(total_hours <= emp['Availability_Hours_per_Week'] * 1.2)  # 20% buffer
    
    # Constraint 4: Priority tasks must be assigned to high-performing employees
    priority_map = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
    for j in range(n_tasks):
        task = task_df.iloc[j]
        if task['Priority'] in ['Critical', 'High']:
            # Sum of performance scores of assigned employees
            for i in range(n_employees):
                emp = employee_df.iloc[i]
                if emp['Performance_1_10'] < 6:
                    # Discourage low-performing employees for high-priority tasks
                    solver.Add(x[i, j] == 0)
    
    print("\nSolving ILP problem...")
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found!")
        
        # Extract assignments
        assignments = []
        
        for i in range(n_employees):
            for j in range(n_tasks):
                if x[i, j].solution_value() > 0.5:
                    emp = employee_df.iloc[i]
                    task = task_df.iloc[j]
                    
                    assignment = {
                        'TaskID': task['TaskID'],
                        'TaskName': task['TaskName'],
                        'AssignedEmployeeID': emp['EmployeeID'],
                        'EmployeeName': emp['Name'],
                        'EmployeeRole': emp['Role'],
                        'Department': task['Department'],
                        'SuitabilityScore': suitability_scores[i, j],
                        'PredictedHours': predicted_hours[i, j],
                        'EstimatedHours': task['Estimated_Hours'],
                        'DeadlineDays': task['Deadline_Days'],
                        'Priority': task['Priority'],
                        'Difficulty': task['Difficulty'],
                        'EmployeeExperience': emp['Experience_Years'],
                        'RequiredExperience': task['Required_Experience'],
                        'EmployeePerformance': emp['Performance_1_10'],
                        'EmployeeWorkload': emp['Current_Workload_Tasks'],
                        'EmployeeAvailability': emp['Availability_Hours_per_Week']
                    }
                    
                    assignments.append(assignment)
        
        assignments_df = pd.DataFrame(assignments)
        
        # Sort by priority and deadline
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        assignments_df['PriorityOrder'] = assignments_df['Priority'].map(priority_order)
        assignments_df = assignments_df.sort_values(['PriorityOrder', 'DeadlineDays'])
        assignments_df = assignments_df.drop('PriorityOrder', axis=1)
        
        # Print summary statistics
        print(f"\nTotal assignments: {len(assignments_df)}")
        print(f"Average suitability score: {assignments_df['SuitabilityScore'].mean():.2f}")
        print(f"Average predicted hours: {assignments_df['PredictedHours'].mean():.2f}")
        
        print(f"\nAssignments by priority:")
        print(assignments_df['Priority'].value_counts().sort_index())
        
        # Workload distribution
        tasks_per_emp = assignments_df.groupby('AssignedEmployeeID').size()
        print(f"\nTasks per employee:")
        print(f"  Min: {tasks_per_emp.min()}")
        print(f"  Max: {tasks_per_emp.max()}")
        print(f"  Mean: {tasks_per_emp.mean():.2f}")
        
        return assignments_df
    
    else:
        print("No optimal solution found!")
        return pd.DataFrame()


def save_assignments(
    assignments_df: pd.DataFrame,
    save_path: str = "../outputs/final_assignments.csv"
):
    """Save final assignments to CSV."""
    print(f"\nSaving assignments to {save_path}...")
    assignments_df.to_csv(save_path, index=False)
    print("Assignments saved successfully!")


def optimize_assignments(
    pairs_df: pd.DataFrame,
    employee_df: pd.DataFrame,
    task_df: pd.DataFrame,
    method: str = 'ilp',
    max_tasks_per_employee: int = 2,
    alpha: float = 0.6,
    beta: float = 0.4
) -> pd.DataFrame:
    """
    Optimize task assignments using specified method.
    
    Args:
        pairs_df: DataFrame with predictions for all employee-task pairs
        employee_df: Employee DataFrame
        task_df: Task DataFrame
        method: 'hungarian' or 'ilp'
        max_tasks_per_employee: Maximum tasks per employee (for ILP)
        alpha: Weight for suitability
        beta: Weight for time efficiency
    
    Returns:
        DataFrame with optimal assignments
    """
    print("\n" + "="*60)
    print("OPTIMIZING TASK ASSIGNMENTS")
    print("="*60)
    
    # Create matrices for optimization
    n_employees = len(employee_df)
    n_tasks = len(task_df)
    
    # Initialize matrices
    suitability_matrix = np.zeros((n_employees, n_tasks))
    hours_matrix = np.zeros((n_employees, n_tasks))
    
    # Fill matrices from pairs_df
    for _, row in pairs_df.iterrows():
        emp_idx = employee_df[employee_df['EmployeeID'] == row['EmployeeID']].index[0]
        task_idx = task_df[task_df['TaskID'] == row['TaskID']].index[0]
        suitability_matrix[emp_idx, task_idx] = row['predicted_suitability']
        hours_matrix[emp_idx, task_idx] = row['predicted_hours']
    
    # Create cost matrix
    cost_matrix = create_cost_matrix(
        employee_df, task_df, suitability_matrix, hours_matrix, alpha, beta
    )
    
    # Optimize based on method
    if method == 'hungarian':
        assignments_df = hungarian_assignment(
            cost_matrix, employee_df, task_df, suitability_matrix, hours_matrix
        )
    else:  # ilp
        assignments_df = ilp_assignment_with_constraints(
            employee_df, task_df, suitability_matrix, hours_matrix,
            max_tasks_per_employee=max_tasks_per_employee,
            alpha=alpha, beta=beta
        )
    
    return assignments_df


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    from feature_engineering import engineer_features_pipeline
    import joblib
    
    # Load data
    employee_df, task_df = preprocess_pipeline(
        "../data/employee_dataset_532.csv",
        "../data/task_dataset_40.csv"
    )
    
    pairs_df, emp_emb, task_emb, sim_matrix = engineer_features_pipeline(
        employee_df, task_df
    )
    
    # Load models
    print("\nLoading trained models...")
    suitability_model = joblib.load("../models/suitability_model.pkl")['model']
    workload_model = joblib.load("../models/workload_model.pkl")['model']
    
    # Predict suitability scores for all pairs
    print("\nPredicting suitability scores...")
    # (Implementation continues in main pipeline)
