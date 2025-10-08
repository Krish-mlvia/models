#!/bin/bash

# Workforce Optimization ML Pipeline - Quick Start Script
# Run this script to set up and execute the entire pipeline

echo "=================================================="
echo "  Workforce Optimization ML Pipeline Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${YELLOW}Step 1: Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
echo ""

# Step 2: Create virtual environment
echo -e "${YELLOW}Step 2: Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Step 3: Activate virtual environment
echo -e "${YELLOW}Step 3: Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 4: Upgrade pip
echo -e "${YELLOW}Step 4: Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Step 5: Install dependencies
echo -e "${YELLOW}Step 5: Installing dependencies (this may take 3-5 minutes)...${NC}"
pip install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

# Step 6: Verify data files
echo -e "${YELLOW}Step 6: Verifying data files...${NC}"
if [ -f "data/employee_dataset_532.csv" ] && [ -f "data/task_dataset_40.csv" ]; then
    echo -e "${GREEN}✓ Data files found${NC}"
    echo "  - employee_dataset_532.csv: $(wc -l < data/employee_dataset_532.csv) lines"
    echo "  - task_dataset_40.csv: $(wc -l < data/task_dataset_40.csv) lines"
else
    echo -e "${RED}✗ Data files missing${NC}"
    exit 1
fi
echo ""

# Step 7: Run the pipeline
echo -e "${YELLOW}Step 7: Running ML Pipeline (this will take 10-15 minutes)...${NC}"
echo "This includes:"
echo "  - Data preprocessing"
echo "  - Feature engineering with embeddings"
echo "  - Training suitability model"
echo "  - Training workload model"
echo "  - Optimizing task assignments"
echo ""
echo "Press ENTER to continue or Ctrl+C to cancel..."
read

cd src
python main_pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================================="
    echo "  ✓ Pipeline completed successfully!"
    echo "==================================================${NC}"
    echo ""
    echo "Results available in:"
    echo "  - models/suitability_model.pkl"
    echo "  - models/workload_model.pkl"
    echo "  - outputs/final_assignments.csv"
    echo "  - outputs/metrics_report.txt"
    echo ""
    echo "To view results:"
    echo "  cd ../outputs"
    echo "  cat metrics_report.txt"
    echo "  head -20 final_assignments.csv"
else
    echo -e "${RED}✗ Pipeline failed. Check error messages above.${NC}"
    exit 1
fi
