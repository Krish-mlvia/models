#!/bin/bash

# ============================================
# COMPLETE STEP-BY-STEP COMMANDS
# Copy and run these in your terminal
# ============================================

echo "Starting Workforce Optimization ML Pipeline..."
echo ""

# Step 1: Navigate to project
cd /Users/krishmalvia/Desktop/pms-models

# Step 2: Activate virtual environment
source venv/bin/activate

# Step 3: Run the complete pipeline
cd src
python run_pipeline_simple.py

echo ""
echo "✅ Complete! Check outputs/ folder for results."
