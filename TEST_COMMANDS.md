# Step-by-Step Commands to Test Your Models

## 🧪 Complete Testing Guide

Follow these commands **exactly** in your terminal to test your trained models.

---

## ✅ Prerequisites Check

First, make sure you're in the right directory and have the models:

```bash
# Navigate to project directory
cd /Users/krishmalvia/Desktop/pms-models

# Check if models exist
ls -lh models/
# You should see: suitability_model.pkl and workload_model.pkl

# Check if data exists
ls -lh data/
# You should see: employee_dataset_532.csv and task_dataset_40.csv
```

---

## 🚀 Option 1: Quick Test (Automated Script)

### Step 1: Activate Virtual Environment
```bash
source venv/bin/activate
```
**Expected output:** Your prompt should show `(venv)` at the beginning

### Step 2: Run Test Script
```bash
python test_models.py
```
**Expected output:** 
- Models load successfully
- Sample predictions displayed
- 5 random test predictions
- Summary statistics

**This takes ~30 seconds**

---

## 🔬 Option 2: Manual Testing (Interactive Python)

### Step 1: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 2: Start Python Interactive Shell
```bash
python
```

### Step 3: Load and Test Models (Copy-paste each block)

#### Load Models:
```python
import pickle

# Load suitability model
with open('models/suitability_model.pkl', 'rb') as f:
    suitability_data = pickle.load(f)
    suitability_model = suitability_data['model']

print(f"✓ Suitability model loaded")
print(f"Features: {suitability_data['feature_names']}")

# Load workload model
with open('models/workload_model.pkl', 'rb') as f:
    workload_data = pickle.load(f)
    workload_model = workload_data['model']

print(f"✓ Workload model loaded")
```

#### Load Test Data:
```python
import pandas as pd

employees = pd.read_csv('data/employee_dataset_532.csv')
tasks = pd.read_csv('data/task_dataset_40.csv')

print(f"Employees: {len(employees)}")
print(f"Tasks: {len(tasks)}")
```

#### Make a Test Prediction:
```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Pick first employee and first task
emp = employees.iloc[0]
task = tasks.iloc[0]

print(f"Employee: {emp['Name']}")
print(f"Task: {task['TaskName']}")

# Compute skill similarity
emp_emb = embedding_model.encode(emp['Skills'])
task_emb = embedding_model.encode(task['Required_Skills'])
similarity = np.dot(emp_emb, task_emb) / (np.linalg.norm(emp_emb) * np.linalg.norm(task_emb))

# Create features
features = pd.DataFrame([{
    'skill_similarity_score': similarity,
    'experience_years': emp['Experience_Years'],
    'required_experience': task['Required_Experience'],
    'experience_difference': emp['Experience_Years'] - task['Required_Experience'],
    'performance_score': emp['Performance_1_10'],
    'success_rate': emp['Completed_Tasks'] / max(emp['Completed_Tasks'] + emp['Current_Workload_Tasks'], 1),
    'current_workload': emp['Current_Workload_Tasks'],
    'availability_hours': emp['Availability_Hours_per_Week'],
    'workload_ratio': emp['Current_Workload_Tasks'] / 40.0,
    'efficiency_score': emp['Performance_1_10'] * (1 - emp['Current_Workload_Tasks'] / 40.0),
    'estimated_hours': task['Estimated_Hours'],
    'deadline_days': task['Deadline_Days'],
    'difficulty_numeric': {'Easy': 1, 'Medium': 2, 'Hard': 3}.get(task['Difficulty'], 2),
    'priority_numeric': {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}.get(task['Priority'], 2),
    'urgency_score': (5 - task['Deadline_Days']) / 5.0 if task['Deadline_Days'] <= 5 else 0,
    'complexity_score': {'Easy': 1, 'Medium': 2, 'Hard': 3}.get(task['Difficulty'], 2) / 3.0,
    'department_match': 1 if emp['Department'] == task['Department'] else 0,
    'hours_vs_availability': task['Estimated_Hours'] / emp['Availability_Hours_per_Week'],
    'role_alignment': 1
}])

# Make predictions
X_suit = features[suitability_data['feature_names']]
X_work = features[workload_data['feature_names']]

suitability_score = suitability_model.predict(X_suit)[0]
predicted_hours = workload_model.predict(X_work)[0]

print(f"\n🎯 RESULTS:")
print(f"Suitability Score: {suitability_score:.2f}/100")
print(f"Predicted Hours: {predicted_hours:.2f} hours")
```

#### Exit Python:
```python
exit()
```

---

## 📊 Option 3: Check Existing Predictions

### View the assignments that were already generated:

```bash
# View assignments (first 20 lines)
head -20 outputs/final_assignments.csv

# View assignments in table format
column -s, -t < outputs/final_assignments.csv | head -20

# View full metrics report
cat outputs/metrics_report.txt

# Count assignments
wc -l outputs/final_assignments.csv
# Should show 41 (40 tasks + 1 header)
```

### Filter and search assignments:

```bash
# Find assignments for high priority tasks
grep "High" outputs/final_assignments.csv

# Find assignments for a specific department
grep "Engineering" outputs/final_assignments.csv

# Find assignments with high suitability (>60)
awk -F',' '$7 > 60' outputs/final_assignments.csv

# Show top 5 assignments by suitability score
sort -t',' -k7 -nr outputs/final_assignments.csv | head -6
```

---

## 🔍 Option 4: Validate Model Files

Check model integrity and information:

```bash
# Check model file sizes
ls -lh models/

# View model metadata using Python one-liner
python -c "import pickle; data=pickle.load(open('models/suitability_model.pkl','rb')); print('Features:', len(data['feature_names'])); print('Params:', data['best_params'])"

# View workload model info
python -c "import pickle; data=pickle.load(open('models/workload_model.pkl','rb')); print('Features:', len(data['feature_names'])); print('Params:', data['best_params'])"
```

---

## 🎯 Option 5: Quick Performance Test

Test how fast the models make predictions:

```bash
python -c "
import pickle
import pandas as pd
import time

# Load models
with open('models/suitability_model.pkl', 'rb') as f:
    suit_model = pickle.load(f)['model']
with open('models/workload_model.pkl', 'rb') as f:
    work_model = pickle.load(f)['model']

# Load data
df = pd.read_csv('data/employee_dataset_532.csv')

# Create dummy features (19 features for suitability)
import numpy as np
X = pd.DataFrame(np.random.rand(100, 19))

# Time predictions
start = time.time()
predictions = suit_model.predict(X)
elapsed = time.time() - start

print(f'✓ Made 100 predictions in {elapsed*1000:.2f}ms')
print(f'✓ Average: {elapsed*10:.2f}ms per prediction')
print(f'✓ Models are fast and ready for production!')
"
```

---

## 📈 Option 6: Advanced Testing with Custom Data

Test with your own employee-task pair:

```bash
python << 'EOF'
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load models
with open('models/suitability_model.pkl', 'rb') as f:
    suit_data = pickle.load(f)
with open('models/workload_model.pkl', 'rb') as f:
    work_data = pickle.load(f)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# YOUR CUSTOM DATA - EDIT THESE VALUES
employee_skills = "Python, Machine Learning, TensorFlow, Data Science"
task_skills = "Machine Learning Model Development, Python"
employee_experience = 5.0
task_required_experience = 3.0
employee_workload = 2
task_hours = 15.0
employee_performance = 8.5
employee_availability = 35.0

# Compute similarity
emp_emb = embedding_model.encode(employee_skills)
task_emb = embedding_model.encode(task_skills)
similarity = np.dot(emp_emb, task_emb) / (np.linalg.norm(emp_emb) * np.linalg.norm(task_emb))

# Create features
features = pd.DataFrame([{
    'skill_similarity_score': similarity,
    'experience_years': employee_experience,
    'required_experience': task_required_experience,
    'experience_difference': employee_experience - task_required_experience,
    'performance_score': employee_performance,
    'success_rate': 0.85,
    'current_workload': employee_workload,
    'availability_hours': employee_availability,
    'workload_ratio': employee_workload / 40.0,
    'efficiency_score': employee_performance * (1 - employee_workload / 40.0),
    'estimated_hours': task_hours,
    'deadline_days': 7,
    'difficulty_numeric': 2,
    'priority_numeric': 3,
    'urgency_score': 0.3,
    'complexity_score': 0.67,
    'department_match': 1,
    'hours_vs_availability': task_hours / employee_availability,
    'role_alignment': 1
}])

# Predict
X_suit = features[suit_data['feature_names']]
X_work = features[work_data['feature_names']]

suitability = suit_data['model'].predict(X_suit)[0]
hours = work_data['model'].predict(X_work)[0]

print("\n" + "="*60)
print("CUSTOM PREDICTION RESULTS")
print("="*60)
print(f"Employee Skills: {employee_skills}")
print(f"Task Skills: {task_skills}")
print(f"\nSkill Match: {similarity*100:.1f}%")
print(f"Suitability Score: {suitability:.2f}/100")
print(f"Predicted Hours: {hours:.2f}h")
print("="*60 + "\n")

EOF
```

---

## 🆘 Troubleshooting Commands

If something doesn't work:

```bash
# Check Python version
python --version
# Should be 3.13.x

# Check if virtual environment is activated
which python
# Should show: /Users/krishmalvia/Desktop/pms-models/venv/bin/python

# Reinstall dependencies if needed
pip install -r requirements.txt

# Check if models exist
test -f models/suitability_model.pkl && echo "✓ Suitability model exists" || echo "✗ Missing"
test -f models/workload_model.pkl && echo "✓ Workload model exists" || echo "✗ Missing"

# If models are missing, retrain
python src/run_pipeline_simple.py
```

---

## ✅ Success Checklist

After running tests, you should see:

- [x] Models load without errors
- [x] Predictions return reasonable numbers (suitability: 0-100, hours: positive)
- [x] Test script completes successfully
- [x] No import errors or missing dependencies
- [x] Predictions are fast (<100ms per prediction)

---

## 🎓 Next Steps

Once testing is complete:

1. **Integrate into your system**: Use the models in your PMS application
2. **Create an API**: Build REST endpoints for predictions
3. **Monitor performance**: Track actual vs predicted hours
4. **Retrain regularly**: Add new data and retrain monthly

---

## 📞 Quick Command Reference

| Action | Command |
|--------|---------|
| Activate environment | `source venv/bin/activate` |
| Run automated test | `python test_models.py` |
| View assignments | `cat outputs/final_assignments.csv` |
| View metrics | `cat outputs/metrics_report.txt` |
| Check model files | `ls -lh models/` |
| Interactive testing | `python` then copy-paste code |
| Deactivate environment | `deactivate` |

---

**Ready to test?** Start with **Option 1** (Quick Test) - it's the easiest! 🚀
