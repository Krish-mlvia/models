# How to Use Your Workforce Optimization Models

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Using Trained Models](#using-trained-models)
3. [Making Predictions](#making-predictions)
4. [Retraining Models](#retraining-models)
5. [Adding New Data](#adding-new-data)

---

## 🚀 Quick Start

### Step 1: Navigate to Project Directory
```bash
cd /Users/krishmalvia/Desktop/pms-models
```

### Step 2: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 3: Run the Complete Pipeline
```bash
python src/run_pipeline_simple.py
```

**That's it!** The pipeline will:
- Load and preprocess data
- Generate skill embeddings
- Train both models
- Optimize task assignments
- Save results to `outputs/`

---

## 🎯 Using Trained Models

### Option A: Use Existing Models (No Retraining)

If you already have trained models and just want to make new assignments:

#### Step 1: Activate Environment
```bash
cd /Users/krishmalvia/Desktop/pms-models
source venv/bin/activate
```

#### Step 2: Create a Prediction Script

Create `src/use_models.py`:

```python
import pickle
import pandas as pd
from feature_engineering import engineer_features_pipeline
from assignment_optimizer import optimize_assignments

# Load trained models
with open('models/suitability_model.pkl', 'rb') as f:
    suitability_model = pickle.load(f)

with open('models/workload_model.pkl', 'rb') as f:
    workload_model = pickle.load(f)

print("✅ Models loaded successfully!")

# Load your data
employees = pd.read_csv('data/employee_dataset_532.csv')
tasks = pd.read_csv('data/task_dataset_40.csv')

# Generate features
print("🔄 Generating features...")
features_df = engineer_features_pipeline(employees, tasks)

# Make predictions
print("🎯 Making predictions...")
X = features_df.drop(['EmployeeID', 'TaskID', 'suitability_score', 
                      'actual_hours'], axis=1, errors='ignore')
suitability_scores = suitability_model.predict(X)
predicted_hours = workload_model.predict(X)

# Add predictions to dataframe
features_df['predicted_suitability'] = suitability_scores
features_df['predicted_hours'] = predicted_hours

# Optimize assignments
print("⚡ Optimizing assignments...")
assignments = optimize_assignments(
    employees, tasks, 
    features_df[['EmployeeID', 'TaskID', 'predicted_suitability', 'predicted_hours']]
)

# Save results
assignments.to_csv('outputs/new_assignments.csv', index=False)
print(f"✅ Saved {len(assignments)} assignments to outputs/new_assignments.csv")
```

#### Step 3: Run Prediction Script
```bash
python src/use_models.py
```

---

## 🔮 Making Predictions

### For a Single Employee-Task Pair

Create `src/predict_single.py`:

```python
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load models
with open('models/suitability_model.pkl', 'rb') as f:
    suitability_model = pickle.load(f)

with open('models/workload_model.pkl', 'rb') as f:
    workload_model = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def predict_assignment(employee_skills, task_skills, 
                       employee_experience, task_experience,
                       employee_workload, task_hours):
    """
    Predict suitability and time for one employee-task pair
    
    Args:
        employee_skills: str, e.g., "Python, Machine Learning, Data Analysis"
        task_skills: str, e.g., "Python, ML Model Development"
        employee_experience: float, years of experience
        task_experience: float, required experience
        employee_workload: int, current number of tasks
        task_hours: float, estimated hours for task
    """
    
    # Generate embeddings
    emp_emb = embedding_model.encode(employee_skills)
    task_emb = embedding_model.encode(task_skills)
    
    # Compute similarity
    similarity = np.dot(emp_emb, task_emb) / (
        np.linalg.norm(emp_emb) * np.linalg.norm(task_emb)
    )
    
    # Create feature vector (simplified - add more features as needed)
    features = pd.DataFrame([{
        'skill_similarity_score': similarity,
        'experience_difference': employee_experience - task_experience,
        'workload_ratio': employee_workload / 40.0,  # normalize
        'estimated_hours': task_hours,
        'efficiency_score': 1.0,  # placeholder
        'complexity_score': 0.5,  # placeholder
        'urgency_score': 0.5,  # placeholder
        # Add other required features...
    }])
    
    # Make predictions
    suitability = suitability_model.predict(features)[0]
    hours = workload_model.predict(features)[0]
    
    return {
        'suitability_score': round(suitability, 2),
        'predicted_hours': round(hours, 2),
        'skill_match': f"{similarity*100:.1f}%"
    }

# Example usage
result = predict_assignment(
    employee_skills="Python, Machine Learning, TensorFlow, Data Science",
    task_skills="Machine Learning Model Development",
    employee_experience=5.0,
    task_experience=3.0,
    employee_workload=2,
    task_hours=15.0
)

print("Prediction Results:")
print(f"  Suitability Score: {result['suitability_score']}/100")
print(f"  Predicted Hours: {result['predicted_hours']} hours")
print(f"  Skill Match: {result['skill_match']}")
```

Run it:
```bash
python src/predict_single.py
```

---

## 🔄 Retraining Models

### When to Retrain:
- New employee data added
- New task completions recorded
- Performance metrics decline
- Every 1-3 months for best results

### Step 1: Update Your Data Files
Add new rows to:
- `data/employee_dataset_532.csv`
- `data/task_dataset_40.csv`

### Step 2: Run Full Pipeline
```bash
cd /Users/krishmalvia/Desktop/pms-models
source venv/bin/activate
python src/run_pipeline_simple.py
```

This will:
- Reload all data
- Retrain both models with new data
- Update `models/suitability_model.pkl`
- Update `models/workload_model.pkl`
- Generate new assignments

### Step 3: Verify Results
Check the metrics in `outputs/metrics_report.txt`:
```bash
cat outputs/metrics_report.txt
```

---

## ➕ Adding New Data

### Adding New Employees

Edit `data/employee_dataset_532.csv` and add rows with these columns:

```csv
EmployeeID,Name,Role,Department,Skills,Experience_Years,Certifications,Performance_1_10,Current_Workload_Tasks,Completed_Tasks,Availability_Hours_per_Week,Preferred_Task_Type,Collaboration_Score_1_10,Remote_Onsite,Previous_Projects
533,John Doe,ML Engineer,Engineering,"Python, TensorFlow, Deep Learning",4,AWS Certified,8.5,2,15,35,Development,7.8,Remote,"Project A, Project B"
```

### Adding New Tasks

Edit `data/task_dataset_40.csv` and add rows:

```csv
TaskID,TaskName,Required_Skills,Estimated_Hours,Deadline_Days,Priority,Difficulty,Required_Experience,Department,Task_Type
41,Build API Endpoint,"Python, Flask, REST API",20,14,High,Medium,3,Engineering,Development
```

### Retrain After Adding Data
```bash
python src/run_pipeline_simple.py
```

---

## 📊 Understanding the Outputs

### 1. Final Assignments (`outputs/final_assignments.csv`)

Contains the optimized task-employee matches:

| Column | Description |
|--------|-------------|
| `TaskID` | Unique task identifier |
| `TaskName` | Name of the task |
| `AssignedEmployeeID` | ID of assigned employee |
| `EmployeeName` | Name of assigned employee |
| `SuitabilityScore` | How well matched (0-100) |
| `PredictedHours` | Estimated completion time |
| `Priority` | Task priority level |

### 2. Metrics Report (`outputs/metrics_report.txt`)

Shows model performance:
- **Suitability Model**: R², RMSE, MAE scores
- **Workload Model**: R², RMSE, MAE in hours
- **Assignment Stats**: Distribution by department, priority

---

## 🛠️ Customization Options

### Adjust Optimization Weights

Edit `src/assignment_optimizer.py`, function `create_cost_matrix`:

```python
# Current: 60% suitability, 40% time
cost_matrix = alpha * (100 - suitability_matrix) + beta * normalized_time_matrix

# More focus on suitability (80/20):
cost_matrix = 0.8 * (100 - suitability_matrix) + 0.2 * normalized_time_matrix

# More focus on time (40/60):
cost_matrix = 0.4 * (100 - suitability_matrix) + 0.6 * normalized_time_matrix
```

### Change Task Limits Per Employee

Edit `src/assignment_optimizer.py`, function `ilp_assignment_with_constraints`:

```python
# Current: Max 2 tasks per employee
for e in range(n_employees):
    solver.Add(solver.Sum([x[e, t] for t in range(n_tasks)]) <= 2)

# Allow up to 5 tasks:
for e in range(n_employees):
    solver.Add(solver.Sum([x[e, t] for t in range(n_tasks)]) <= 5)
```

### Adjust Hyperparameter Search

Edit `src/train_suitability_model.py` or `src/train_workload_model.py`:

```python
# Current: 30 trials
study.optimize(objective, n_trials=30)

# More thorough search (slower):
study.optimize(objective, n_trials=100)

# Faster search:
study.optimize(objective, n_trials=10)
```

---

## 🐛 Troubleshooting

### Models Not Found
```bash
# Train models first:
python src/run_pipeline_simple.py
```

### Import Errors
```bash
# Reinstall dependencies:
source venv/bin/activate
pip install -r requirements.txt
```

### Poor Predictions
```bash
# Retrain with more data or check data quality:
python src/run_pipeline_simple.py
```

### Memory Issues
```bash
# Reduce batch size in feature_engineering.py
# Or use a machine with more RAM
```

---

## 📞 Quick Reference Commands

| Task | Command |
|------|---------|
| Activate environment | `source venv/bin/activate` |
| Run full pipeline | `python src/run_pipeline_simple.py` |
| View assignments | `cat outputs/final_assignments.csv` |
| View metrics | `cat outputs/metrics_report.txt` |
| Check model files | `ls -lh models/` |
| Deactivate environment | `deactivate` |

---

## 🎓 Next Steps

1. **Production Integration**: Export predictions to your PMS system
2. **API Development**: Build REST API for real-time predictions
3. **Monitoring**: Track actual vs predicted hours for continuous improvement
4. **Automation**: Schedule weekly retraining with cron jobs
5. **Dashboard**: Visualize assignments and metrics with Streamlit or Dash

---

**Need Help?** Check the other documentation files:
- `SETUP_GUIDE.md` - Initial setup
- `COMMANDS_REFERENCE.md` - All available commands
- `PROJECT_SUMMARY.md` - Technical architecture
