#!/usr/bin/env python3
"""
Example: Compare calibration WITH and WITHOUT time stretching
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

def run_with_time_stretch(time_stretch_factor):
    """Run calibration with specified time stretch factor"""
    
    calib_config = CalibrationConfig(
        T=180,
        timesteps_per_day=4,
        mode="BETA_ONLY",
        optimizers=["L-BFGS-B"],
        verbose_lbfgs=False,
        verbosity=2,
        loss_aggregation="regional",
        noise_type="poisson",
        apply_time_stretch=True if time_stretch_factor > 1.0 else False,
        time_stretch_factor=time_stretch_factor
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
    Compare calibration results with different time stretching factors
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = start_logging(f"time_stretching_log_{timestamp}.txt")
    
    try:
    
        print("="*100)
        print("TIME STRETCHING COMPARISON EXAMPLE")
        print("="*100)
        print("\nTesting three scenarios:")
        print("  1. No time stretching (factor = 1.0)")
        print("  2. Moderate stretching (factor = 3.0)")
        print("  3. Professor's stretching (factor = 5.0)")
        print("="*100)
        
        scenarios = [
            ("No Stretching", 1.0),
            ("Moderate Stretching", 3.0),
            ("Professor's Stretching", 5.0)
        ]
        
        results_summary = []
        
        for scenario_name, factor in scenarios:
            print(f"\n{'#'*100}")
            print(f"# SCENARIO: {scenario_name} (factor = {factor})")
            print(f"{'#'*100}")
            
            stage1_results, _ = run_with_time_stretch(factor)
            
            # Extract best result
            best = min(stage1_results.items(), key=lambda x: x[1]['loss'])
            best_opt, best_result = best
            
            results_summary.append({
                'scenario': scenario_name,
                'factor': factor,
                'optimizer': best_opt,
                'loss': best_result['loss'],
                'r2': best_result.get('global_r2', 0.0),
                'offset': best_result.get('offset', 0)
            })
        
        # Print comparison summary
        print("\n" + "="*100)
        print("TIME STRETCHING COMPARISON SUMMARY")
        print("="*100)
        print(f"{'SCENARIO':<30} | {'FACTOR':<10} | {'LOSS':<12} | {'R²':<10} | {'OFFSET':<10}")
        print("-"*100)
        
        for result in results_summary:
            print(f"{result['scenario']:<30} | {result['factor']:<10.1f} | "
                f"{result['loss']:<12.4f} | {result['r2']:<10.6f} | {result['offset']:<10}")
        
        print("="*100)
        
        # Analysis
        best_scenario = min(results_summary, key=lambda x: x['loss'])
        print(f"\nBest scenario: {best_scenario['scenario']} "
            f"(Loss: {best_scenario['loss']:.4f}, R²: {best_scenario['r2']:.6f})")
        
        print("\nConclusion:")
        if best_scenario['factor'] > 1.0:
            print(f"  Time stretching improves calibration quality.")
            print(f"  Recommended factor: {best_scenario['factor']}")
        else:
            print(f"  No time stretching performs best for this scenario.")
            print(f"  Time stretching may not be necessary for all applications.")
    
    finally:
        stop_logging(logger)

if __name__ == "__main__":
    main()