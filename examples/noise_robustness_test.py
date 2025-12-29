#!/usr/bin/env python3
"""
Example: Test calibration robustness to different noise types
"""

import sys
from pathlib import Path
from datetime import datetime
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.calibration_config import CalibrationConfig, RegularizationConfig
from config.model_config import ModelConfig
from scripts.run_calibration import run_calibration
from src.utils.logger import start_logging, stop_logging

def run_with_noise_type(noise_type, noise_seed):
    """Run calibration with specified noise type"""
    
    calib_config = CalibrationConfig(
        T=180,
        timesteps_per_day=4,
        mode="BETA_ONLY",
        optimizers=["L-BFGS-B", "CG"],
        verbose_lbfgs=False,
        verbosity=2,
        loss_aggregation="regional",
        noise_type=noise_type,
        noise_seed=noise_seed
    )
    
    calib_config.estimation_config = {
        "beta_param": "L",
        "estimate_initial": {
            "E": True,
            "IP": False,
            "ISR": False,
            "ISH": False,
            "IA": False
        }
    }
    
    calib_config.regularization = RegularizationConfig(
        beta_type="l2_magnitude",
        beta_lambda=1e-2,
        compartment_configs={
            "E": {
                "type": "structural",
                "location_targets": [0.0, 1.0, 0.0],
                "age_targets": [0, 0, 1, 0, 0],
                "lambda_on_target": 10.0,
                "lambda_off_target": 10.0
            }
        }
    )
    
    model_config = ModelConfig()
    
    return run_calibration(calib_config, model_config)

def main():
    """
    Test calibration robustness across different noise types
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = start_logging(f"noise_robustness_log_{timestamp}.txt")
    
    try:
        print("="*100)
        print("NOISE ROBUSTNESS TEST")
        print("="*100)
        print("\nTesting three noise scenarios:")
        print("  1. No noise (clean data)")
        print("  2. Poisson noise (count-based)")
        print("  3. Gaussian noise (continuous)")
        print("="*100)
        
        noise_scenarios = [
            ("Clean", "none", 0),
            ("Poisson", "poisson", 12345),
            ("Gaussian", "gaussian", 12345)
        ]
        
        results_summary = []
        
        for scenario_name, noise_type, seed in noise_scenarios:
            print(f"\n{'#'*100}")
            print(f"# NOISE SCENARIO: {scenario_name}")
            print(f"{'#'*100}")
            
            stage1_results, _ = run_with_noise_type(noise_type, seed)
            
            # Store results for both optimizers
            for opt_name, result in stage1_results.items():
                results_summary.append({
                    'scenario': scenario_name,
                    'noise_type': noise_type,
                    'optimizer': opt_name,
                    'loss': result['loss'],
                    'r2': result.get('global_r2', 0.0)
                })
        
        # Print comparison summary
        print("\n" + "="*100)
        print("NOISE ROBUSTNESS COMPARISON")
        print("="*100)
        print(f"{'SCENARIO':<15} | {'OPTIMIZER':<15} | {'LOSS':<12} | {'R²':<10}")
        print("-"*100)
        
        for result in results_summary:
            print(f"{result['scenario']:<15} | {result['optimizer']:<15} | "
                f"{result['loss']:<12.4f} | {result['r2']:<10.6f}")
        
        print("="*100)
        
        # Analysis by optimizer
        print("\nPer-Optimizer Analysis:")
        for opt_name in ["L-BFGS-B", "CG"]:
            print(f"\n{opt_name}:")
            opt_results = [r for r in results_summary if r['optimizer'] == opt_name]
            
            for result in opt_results:
                print(f"  {result['scenario']:<15}: Loss={result['loss']:.4f}, R²={result['r2']:.6f}")
            
            # Compute robustness (variance in loss)
            losses = [r['loss'] for r in opt_results]
            import numpy as np
            loss_std = np.std(losses)
            print(f"  Loss std dev: {loss_std:.4f} (lower = more robust)")
        
        print("\nConclusion:")
        print("  Lower std dev indicates better robustness to noise.")
        print("  Poisson noise is most realistic for count data (hospital admissions).")
    
    finally:
        stop_logging(logger)

if __name__ == "__main__":
    main()