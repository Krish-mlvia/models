#!/bin/bash
# EXACT STEP-BY-STEP COMMANDS TO RUN YOUR ML MODELS
# Copy and paste these commands one by one into your terminal

echo "=========================================="
echo "STEP-BY-STEP EXECUTION GUIDE"
echo "=========================================="
echo ""
echo "Copy and paste each command block below:"
echo ""

cat << 'EOF'

# ============================================================
# STEP 1: Navigate to Project Directory
# ============================================================
cd /Users/krishmalvia/Desktop/pms-models


# ============================================================
# STEP 2: Create Virtual Environment
# ============================================================
python3 -m venv venv


# ============================================================
# STEP 3: Activate Virtual Environment
# ============================================================
source venv/bin/activate


# ============================================================
# STEP 4: Upgrade pip
# ============================================================
pip install --upgrade pip


# ============================================================
# STEP 5: Install Dependencies (takes 3-5 minutes)
# ============================================================
pip install -r requirements.txt


# ============================================================
# STEP 6: Verify Data Files
# ============================================================
ls -lh data/
# You should see:
# - employee_dataset_532.csv
# - task_dataset_40.csv


# ============================================================
# STEP 7: Test Data Preprocessing
# ============================================================
cd src
python data_preprocessing.py


# ============================================================
# STEP 8: Test Feature Engineering (takes 2-3 minutes)
# ============================================================
python feature_engineering.py


# ============================================================
# STEP 9: Train Suitability Model (takes 5-10 minutes)
# ============================================================
python train_suitability_model.py


# ============================================================
# STEP 10: Train Workload Model (takes 3-5 minutes)
# ============================================================
python train_workload_model.py


# ============================================================
# STEP 11: Run Complete Pipeline (ALL-IN-ONE)
# ============================================================
python main_pipeline.py


# ============================================================
# STEP 12: View Results
# ============================================================
cd ../outputs
ls -lh

# View metrics report
cat metrics_report.txt

# View first 20 assignments
head -20 final_assignments.csv


# ============================================================
# OPTIONAL: Run Automated Script (Alternative to Steps 1-12)
# ============================================================
cd /Users/krishmalvia/Desktop/pms-models
./run_pipeline.sh

EOF
