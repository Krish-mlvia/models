# Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd pms-models

# Install required packages
pip install -r requirements.txt
```

## Quick Run (Automated Pipeline)

```bash
# Run complete pipeline with default settings
python run_pipeline.py
```

This will:
- ✅ Load and preprocess data
- ✅ Generate skill embeddings using Sentence Transformers
- ✅ Train suitability model with hyperparameter tuning (30 trials)
- ✅ Train workload model with hyperparameter tuning (30 trials)
- ✅ Optimize task assignments using ILP
- ✅ Generate visualizations and metrics report

**Estimated time**: 10-20 minutes (depending on hardware)

## Step-by-Step Execution

### 1. Data Preprocessing Only
```bash
cd src
python data_preprocessing.py
```

### 2. Feature Engineering
```bash
python feature_engineering.py
```

### 3. Train Suitability Model
```bash
python train_suitability_model.py
```

### 4. Train Workload Model
```bash
python train_workload_model.py
```

### 5. Optimize Assignments
```bash
python assignment_optimizer.py
```

### 6. Generate Visualizations
```bash
python visualizations.py
```

## Using Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open full_pipeline.ipynb
# Execute cells sequentially
```

## Configuration Options

Edit `run_pipeline.py` or `src/main_pipeline.py` to adjust:

```python
run_complete_pipeline(
    use_optuna=True,           # Enable/disable hyperparameter tuning
    n_trials=30,               # Number of Optuna trials (more = better but slower)
    assignment_method='ilp',    # 'hungarian' for 1:1 or 'ilp' for multi-task
    max_tasks_per_employee=2,  # Maximum tasks per employee (ILP only)
    random_state=42            # Random seed for reproducibility
)
```

## Output Files

After execution, check:

1. **models/suitability_model.pkl** - Trained suitability prediction model
2. **models/workload_model.pkl** - Trained workload prediction model  
3. **outputs/final_assignments.csv** - Optimal task assignments
4. **outputs/metrics_report.txt** - Comprehensive performance metrics
5. **outputs/*.png** - Visualization plots

## Troubleshooting

### Out of Memory
- Reduce `n_trials` to 10-20
- Use smaller sample in feature engineering

### Slow Execution
- Set `use_optuna=False` for faster training with default parameters
- Reduce number of employee-task pairs in feature engineering

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### CUDA/GPU Issues
LightGBM will automatically use CPU if GPU is not available. No configuration needed.

## Expected Performance

### Suitability Model
- **R² Score**: > 0.85
- **NDCG@5**: > 0.90
- **Test RMSE**: < 10.0

### Workload Model
- **R² Score**: > 0.85
- **MAE**: < 1.5 hours
- **MAPE**: < 15%

### Assignments
- Average suitability score: > 70
- Fair workload distribution
- All tasks assigned optimally

## Next Steps

1. Review `outputs/final_assignments.csv`
2. Check `outputs/metrics_report.txt` for detailed performance
3. Open `full_pipeline.ipynb` for interactive analysis
4. Customize features in `src/feature_engineering.py`
5. Adjust optimization constraints in `src/assignment_optimizer.py`

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review code comments in src/ directory
- Ensure all dependencies are installed correctly
