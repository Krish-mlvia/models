# Workforce Optimization ML System

## Project Overview

This project implements a comprehensive machine learning pipeline for **Time and Resource Allocation in Project Management Systems**. It automates skill-based task assignment and workload prediction for software employees using advanced ML techniques.

## 🎯 Objectives

1. **Skill-Based Matching Model**: Predict the best employee for each task based on skill matching, experience, and workload
2. **Workload Prediction Model**: Predict task completion time for specific employees
3. **Optimal Assignment**: Use optimization algorithms to assign tasks efficiently

## 📊 Datasets

- **employee_dataset_532.csv**: 532 employees with skills, experience, performance, workload, and availability
- **task_dataset_40.csv**: 40 tasks with required skills, difficulty, deadlines, and priorities

## 🏗️ Architecture

### 1. Data Preprocessing
- Data cleaning and validation
- Feature engineering
- Derived metrics calculation

### 2. Feature Engineering
- **Sentence Transformers** (all-mpnet-base-v2) for skill embeddings
- Cosine similarity computation for skill matching
- 19+ engineered features including:
  - `skill_similarity_score`
  - `experience_difference`
  - `workload_ratio`
  - `efficiency_score`
  - `urgency_score`
  - `department_match`
  - `role_alignment`

### 3. ML Models

#### Suitability Model (LightGBM)
- **Purpose**: Predict employee-task suitability score (0-100)
- **Features**: 19 engineered features
- **Optimization**: Optuna hyperparameter tuning
- **Evaluation**: RMSE, MAE, R², NDCG@5, Precision@70, ROC-AUC

#### Workload Model (LightGBM Regressor)
- **Purpose**: Predict task completion time (hours)
- **Features**: 16 task and employee features
- **Optimization**: Optuna hyperparameter tuning
- **Evaluation**: MAE, RMSE, R², MAPE

### 4. Assignment Optimization

#### Hungarian Algorithm
- One-to-one task-employee mapping
- Fast and optimal for simple scenarios

#### Integer Linear Programming (ILP)
- Multi-task assignment per employee
- Constraints:
  - Each task assigned to exactly one employee
  - Maximum tasks per employee
  - Respect availability hours
  - Priority task performance requirements

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd pms-models

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
cd src
python main_pipeline.py
```

### Run Individual Modules

```bash
# Data preprocessing only
python data_preprocessing.py

# Feature engineering
python feature_engineering.py

# Train suitability model
python train_suitability_model.py

# Train workload model
python train_workload_model.py

# Generate visualizations
python visualizations.py
```

## 📁 Project Structure

```
pms-models/
├── data/
│   ├── employee_dataset_532.csv
│   └── task_dataset_40.csv
├── models/
│   ├── suitability_model.pkl
│   └── workload_model.pkl
├── outputs/
│   ├── final_assignments.csv
│   ├── metrics_report.txt
│   ├── suitability_feature_importance.png
│   ├── workload_feature_importance.png
│   └── assignment_statistics.png
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_suitability_model.py
│   ├── train_workload_model.py
│   ├── assignment_optimizer.py
│   ├── visualizations.py
│   └── main_pipeline.py
├── full_pipeline.ipynb
├── requirements.txt
└── README.md
```

## 📈 Model Performance

### Suitability Model
- **Target**: R² > 0.85, NDCG@5 > 0.90
- **Metrics**: RMSE, MAE, Precision@70, NDCG

### Workload Model
- **Target**: MAE < 1.5 hours, R² > 0.85
- **Metrics**: MAE, RMSE, MAPE, Accuracy within ±1/2/3 hours

## 🔧 Configuration

Edit parameters in `main_pipeline.py`:

```python
run_complete_pipeline(
    use_optuna=True,           # Use hyperparameter tuning
    n_trials=30,               # Number of Optuna trials
    assignment_method='ilp',    # 'hungarian' or 'ilp'
    max_tasks_per_employee=2,  # Max tasks per employee (ILP only)
    random_state=42            # Random seed
)
```

## 📊 Output Files

1. **suitability_model.pkl**: Trained suitability prediction model
2. **workload_model.pkl**: Trained workload prediction model
3. **final_assignments.csv**: Optimal employee-task assignments
4. **metrics_report.txt**: Comprehensive performance metrics
5. **Visualization plots**: Feature importance and assignment statistics

## 🎓 Key Technologies

- **pandas, numpy**: Data manipulation
- **sentence-transformers**: Skill embeddings
- **scikit-learn**: ML utilities
- **LightGBM**: Gradient boosting models
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model interpretation
- **scipy**: Hungarian algorithm
- **OR-Tools**: Integer linear programming
- **matplotlib, seaborn**: Visualizations

## 📝 Citation

```
Workforce Optimization ML System
Time and Resource Allocation in Project Management
Author: ML Engineering Team
Year: 2025
```

## 🤝 Contributing

This is a complete end-to-end ML pipeline for workforce optimization. For improvements or extensions, consider:
- Adding deep learning models (Transformers)
- Real-time assignment updates
- Multi-objective optimization
- Fairness constraints
- Historical learning from actual completion times

## 📄 License

MIT License

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This system is designed for automated workforce optimization in software project management environments. Adjust parameters and constraints based on your specific organizational needs.
