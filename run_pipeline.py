#!/usr/bin/env python3
"""
Quick runner script for the workforce optimization pipeline.
"""

import sys
sys.path.insert(0, 'src')

from main_pipeline import run_complete_pipeline

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       WORKFORCE OPTIMIZATION ML PIPELINE - Quick Start        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("Running complete ML pipeline...")
    print("This will:")
    print("  1. Preprocess data")
    print("  2. Generate skill embeddings")
    print("  3. Train suitability model (with Optuna)")
    print("  4. Train workload model (with Optuna)")
    print("  5. Optimize task assignments")
    print("  6. Generate visualizations and reports")
    print()
    print("Estimated time: 10-20 minutes (depends on hardware)")
    print("="*70)
    
    # Run pipeline
    final_assignments = run_complete_pipeline(
        use_optuna=True,          # Use hyperparameter tuning
        n_trials=30,              # Number of Optuna trials
        assignment_method='ilp',   # Use Integer Linear Programming
        max_tasks_per_employee=2,  # Max 2 tasks per employee
        random_state=42            # For reproducibility
    )
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 models/suitability_model.pkl")
    print("  📊 models/workload_model.pkl")
    print("  📄 outputs/final_assignments.csv")
    print("  📄 outputs/metrics_report.txt")
    print("  📊 Visualization plots in outputs/")
    print()
    print("Next steps:")
    print("  - Review final_assignments.csv for task assignments")
    print("  - Check metrics_report.txt for model performance")
    print("  - Open full_pipeline.ipynb for detailed analysis")
    print()
