"""
Feature Engineering Module
Creates advanced features including skill embeddings and similarity scores.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class SkillEmbedder:
    """Generate embeddings for skills using sentence transformers."""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize the skill embedder.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully")
    
    def embed_skills(self, skills_list: List[str]) -> np.ndarray:
        """
        Convert skills to embeddings.
        
        Args:
            skills_list: List of skill strings
        
        Returns:
            Array of embeddings
        """
        # Clean and prepare skills
        cleaned_skills = [str(s).strip() if pd.notna(s) else "" for s in skills_list]
        
        # Generate embeddings
        embeddings = self.model.encode(
            cleaned_skills,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings


def compute_skill_similarity(
    employee_embeddings: np.ndarray,
    task_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between employee and task skill embeddings.
    
    Args:
        employee_embeddings: Employee skill embeddings (n_employees, embedding_dim)
        task_embeddings: Task skill embeddings (n_tasks, embedding_dim)
    
    Returns:
        Similarity matrix (n_employees, n_tasks)
    """
    print("\nComputing skill similarity matrix...")
    similarity_matrix = cosine_similarity(employee_embeddings, task_embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix


def create_employee_task_pairs(
    employee_df: pd.DataFrame,
    task_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_k: int = 50
) -> pd.DataFrame:
    """
    Create training pairs of employees and tasks with engineered features.
    
    Args:
        employee_df: Employee DataFrame
        task_df: Task DataFrame
        similarity_matrix: Skill similarity matrix
        top_k: Number of top candidates to consider per task
    
    Returns:
        DataFrame with employee-task pairs and features
    """
    print(f"\nCreating employee-task pairs (top {top_k} per task)...")
    
    pairs = []
    
    for task_idx, task in task_df.iterrows():
        # Get top-k employees by skill similarity
        task_similarities = similarity_matrix[:, task_idx]
        top_employee_indices = np.argsort(task_similarities)[-top_k:][::-1]
        
        for emp_idx in top_employee_indices:
            emp = employee_df.iloc[emp_idx]
            
            # Create feature dictionary
            pair = {
                'EmployeeID': emp['EmployeeID'],
                'TaskID': task['TaskID'],
                'skill_similarity_score': task_similarities[emp_idx],
                'experience_years': emp['Experience_Years'],
                'required_experience': task['Required_Experience'],
                'experience_difference': emp['Experience_Years'] - task['Required_Experience'],
                'performance_score': emp['Performance_1_10'],
                'success_rate': emp['LastProjectSuccessRate'],
                'current_workload': emp['Current_Workload_Tasks'],
                'availability_hours': emp['Availability_Hours_per_Week'],
                'workload_ratio': emp['workload_ratio'],
                'efficiency_score': emp['efficiency_score'],
                'estimated_hours': task['Estimated_Hours'],
                'deadline_days': task['Deadline_Days'],
                'difficulty_numeric': task['Difficulty_Numeric'],
                'priority_numeric': task['Priority_Numeric'],
                'urgency_score': task['urgency_score'],
                'complexity_score': task['complexity_score'],
                'department_match': int(emp['Department'] == task['Department']),
                'hours_vs_availability': task['Estimated_Hours'] / emp['Availability_Hours_per_Week'],
                'role_alignment': 0  # Will be computed based on role matching
            }
            
            # Role alignment scoring
            role_keywords = {
                'Backend Developer': ['backend', 'api', 'database', 'server'],
                'Frontend Developer': ['frontend', 'ui', 'css', 'html', 'react', 'angular'],
                'Data Engineer': ['data', 'etl', 'pipeline', 'analytics'],
                'Machine Learning Engineer': ['ml', 'model', 'ai', 'prediction', 'tensorflow', 'pytorch'],
                'DevOps Engineer': ['devops', 'ci/cd', 'kubernetes', 'docker', 'deployment'],
                'Fullstack Developer': ['fullstack', 'full-stack', 'api', 'ui'],
                'QA Engineer': ['qa', 'test', 'quality', 'bug'],
                'Mobile': ['mobile', 'android', 'ios']
            }
            
            task_name_lower = task['TaskName'].lower()
            task_desc_lower = task['Description'].lower()
            role = emp['Role']
            
            if role in role_keywords:
                keywords = role_keywords[role]
                alignment_score = sum(
                    1 for keyword in keywords 
                    if keyword in task_name_lower or keyword in task_desc_lower
                )
                pair['role_alignment'] = min(alignment_score / len(keywords), 1.0)
            
            # Compute target: suitability score (normalized combination of factors)
            # This will be our training target
            base_suitability = (
                0.30 * pair['skill_similarity_score'] +
                0.20 * min(pair['performance_score'] / 10, 1.0) +
                0.15 * pair['success_rate'] +
                0.10 * min(pair['experience_years'] / 10, 1.0) +
                0.10 * (1 - min(pair['workload_ratio'] / 2, 1.0)) +
                0.10 * pair['department_match'] +
                0.05 * pair['role_alignment']
            )
            
            # Penalty for mismatched experience or overload
            if pair['experience_difference'] < -2:
                base_suitability *= 0.7
            if pair['workload_ratio'] > 1.5:
                base_suitability *= 0.8
            
            pair['suitability_score'] = min(base_suitability * 100, 100)
            
            pairs.append(pair)
    
    pairs_df = pd.DataFrame(pairs)
    print(f"Created {len(pairs_df)} employee-task pairs")
    
    return pairs_df


def engineer_features_pipeline(
    employee_df: pd.DataFrame,
    task_df: pd.DataFrame,
    model_name: str = 'all-mpnet-base-v2'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete feature engineering pipeline.
    
    Args:
        employee_df: Preprocessed employee DataFrame
        task_df: Preprocessed task DataFrame
        model_name: Name of sentence transformer model
    
    Returns:
        Tuple of (pairs_df, employee_embeddings, task_embeddings, similarity_matrix)
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Initialize embedder
    embedder = SkillEmbedder(model_name)
    
    # Generate embeddings
    print("\nGenerating employee skill embeddings...")
    employee_embeddings = embedder.embed_skills(employee_df['Skills'].tolist())
    
    print("\nGenerating task required skill embeddings...")
    task_embeddings = embedder.embed_skills(task_df['Required_Skills'].tolist())
    
    # Compute similarity
    similarity_matrix = compute_skill_similarity(employee_embeddings, task_embeddings)
    
    # Create training pairs
    pairs_df = create_employee_task_pairs(employee_df, task_df, similarity_matrix)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Pairs dataset shape: {pairs_df.shape}")
    print(f"Feature columns: {pairs_df.columns.tolist()}")
    
    return pairs_df, employee_embeddings, task_embeddings, similarity_matrix


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    
    # Load and preprocess data
    employee_df, task_df = preprocess_pipeline(
        "../data/employee_dataset_532.csv",
        "../data/task_dataset_40.csv"
    )
    
    # Engineer features
    pairs_df, emp_emb, task_emb, sim_matrix = engineer_features_pipeline(
        employee_df, task_df
    )
    
    print("\nPairs DataFrame sample:")
    print(pairs_df.head(10))
    print("\nSuitability score statistics:")
    print(pairs_df['suitability_score'].describe())
