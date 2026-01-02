#Example: Multi-compartment calibration with multiple optimizers

import sys
from pathlib import Path
from datetime import datetime
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.calibration_config import CalibrationConfig, RegularizationConfig
from config.model_config import ModelConfig
from scripts.run_calibration import run_calibration
from src.utils.logger import start_logging, stop_logging

def main():
    """
    Multi-compartment example:
    - Estimate Beta + E0 + IP0
    - Multiple optimizers: L-BFGS-B, CG, Adam
    - Restart strategy with warm-starting
    - Structured regularization on E0, L2 on IP0
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = start_logging(f"multi_compartment_log_{timestamp}.txt")
    
    try:
        # Create configuration
        calib_config = CalibrationConfig(
            T=180,
            timesteps_per_day=4,
            mode="BETA_ONLY",  # Not IHR mode, just beta + initial compartments
            optimizers=["L-BFGS-B", "CG"],  # Test multiple optimizers
            verbose_lbfgs=False,
            verbosity=2,
            loss_aggregation="regional",
            noise_type="poisson",
            early_stop_r2=0.9,
            early_stop_patience=10
        )
        
        # Estimation config: Beta + E0 + IP0
        calib_config.estimation_config = {
            "beta_param": "L",
            "estimate_initial": {
                "E": True,   # Estimate E0
                "IP": True,  # Estimate IP0
                "ISR": False,
                "ISH": False,
                "IA": False
            }
        }
        
        # Restart strategy
        calib_config.num_wide_restarts = 2
        calib_config.num_medium_restarts = 2
        calib_config.num_narrow_restarts = 1
        
        # Mixed regularization
        calib_config.regularization = RegularizationConfig(
            beta_type="l2_magnitude",
            beta_lambda=1e-6,
            compartment_configs={
                "E": {
                    "type": "structural",
                    "location_targets": [0.0, 1.0, 0.0],
                    "age_targets": [0, 0, 1, 0, 0],
                    "lambda_on_target": 100000.0,  
                    "lambda_off_target": 100000.0  
                },
                "IP": {
                    "type": "l2_magnitude",
                    "lambda": 1e-8 
                }
            }
        )
        
        model_config = ModelConfig(
            seed_region_idx=1,
            seed_age_idx=2,
            seed_value=1.0
        )
        
        print("="*100)
        print("MULTI-COMPARTMENT CALIBRATION EXAMPLE")
        print("="*100)
        print(f"Mode: {calib_config.mode}")
        print(f"Optimizers: {calib_config.optimizers}")
        print(f"Estimating: Beta + E0 + IP0")
        print(f"Regularization:")
        print(f"  - Beta: L2 magnitude (λ={calib_config.regularization.beta_lambda})")
        print(f"  - E0: Structured (location-age specific)")
        print(f"  - IP0: L2 magnitude (λ=1e-3)")
        print(f"Restart strategy: Wide({calib_config.num_wide_restarts}) → "
            f"Medium({calib_config.num_medium_restarts}) → "
            f"Narrow({calib_config.num_narrow_restarts})")
        print(f"Noise: {calib_config.noise_type}")
        print("="*100)
        
        # Run calibration
        stage1_results, _ = run_calibration(calib_config, model_config)
        
        print("\nMulti-compartment example complete!")
        print("\nGenerated files:")
        print("  - stage1_optimizer_comparison.png")
        print("  - stage1_parameter_recovery.png")
        print("  - stage1_calibration_regional_aggregate.png")
        print("  - calibration_15_panel.png")
    
    finally:
        stop_logging(logger)

if __name__ == "__main__":
    main()
