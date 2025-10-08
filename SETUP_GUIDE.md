# Workforce Optimization ML Pipeline
# Step-by-Step Setup and Execution Guide

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager
- Terminal access

---

## 🚀 STEP-BY-STEP COMMANDS

### STEP 1: Set Up Python Virtual Environment
```bash
# Navigate to project directory
cd /Users/krishmalvia/Desktop/pms-models

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### STEP 2: Install Required Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### STEP 3: Verify Data Files
```bash
# Check if data files exist
ls -lh data/

# Expected output:
# employee_dataset_532.csv
# task_dataset_40.csv
```

### STEP 4: Test Data Preprocessing
```bash
# Run preprocessing module
cd src
python data_preprocessing.py
```

### STEP 5: Test Feature Engineering
```bash
# Run feature engineering (may take 2-3 minutes)
python feature_engineering.py
```

### STEP 6: Train Suitability Model
```bash
# Train the skill-based matching model (takes 5-10 minutes)
python train_suitability_model.py
```

### STEP 7: Train Workload Model
```bash
# Train the workload prediction model (takes 3-5 minutes)
python train_workload_model.py
```

### STEP 8: Run Complete Pipeline
```bash
# Execute the full pipeline end-to-end
python main_pipeline.py
```

### STEP 9: Check Results
```bash
# Navigate to outputs directory
cd ../outputs

# View final assignments
cat final_assignments.csv

# View metrics report
cat metrics_report.txt
```

---

## 📊 Expected Outputs

After running the complete pipeline, you should see:

### In `models/` directory:
- `suitability_model.pkl` - Trained skill matching model
- `workload_model.pkl` - Trained workload prediction model

### In `outputs/` directory:
- `final_assignments.csv` - Optimal task assignments
- `metrics_report.txt` - Performance metrics
- `suitability_predictions.csv` - All suitability predictions
- `workload_predictions.csv` - All workload predictions
- `feature_importance.png` - Feature importance plots
- `shap_summary.png` - SHAP value visualizations

---

## 🔧 Troubleshooting

### If you get import errors:
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### If memory issues occur:
```bash
# Reduce n_trials in training scripts
# Edit train_suitability_model.py and train_workload_model.py
# Change n_trials from 50 to 20
```

### If sentence-transformers fails:
```bash
# Install specific version
pip install sentence-transformers==2.2.2 --no-cache-dir
```

---

## ⚡ Quick Start (All-in-One)
```bash
# Run everything at once
cd /Users/krishmalvia/Desktop/pms-models
source venv/bin/activate
cd src
python main_pipeline.py
```

---

## 📈 Performance Targets

### Suitability Model:
- Test R² > 0.85
- Test RMSE < 5.0
- Test MAE < 3.5

### Workload Model:
- Test R² > 0.85
- Test RMSE < 2.0 hours
- Test MAE < 1.5 hours

---

## 🎯 Model Usage

### Load and Use Suitability Model:
```python
import joblib
model_pkg = joblib.load('../models/suitability_model.pkl')
model = model_pkg['model']
predictions = model.predict(X_test)
```

### Load and Use Workload Model:
```python
import joblib
model_pkg = joblib.load('../models/workload_model.pkl')
model = model_pkg['model']
time_predictions = model.predict(X_test)
```

---

## 📝 Notes
- First run may take 10-15 minutes due to model downloads
- Subsequent runs will be faster (5-8 minutes)
- Ensure at least 4GB RAM available
- GPU not required but can speed up training

---

## ✅ Success Indicators
You'll know it worked when you see:
1. "✅ Suitability model training complete!"
2. "✅ Workload model training complete!"
3. "✅ Assignment optimization complete!"
4. Files created in outputs/ directory

---

## 🆘 Support
If issues persist, check:
1. Python version: `python3 --version` (should be 3.8+)
2. Pip version: `pip --version`
3. Virtual environment is activated
4. All dependencies installed: `pip list`
