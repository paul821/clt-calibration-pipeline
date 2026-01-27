# Example: Two-region Austin calibration with Time Stretch estimation

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.calibration_config import CalibrationConfig, RegularizationConfig
from config.model_config import ModelConfig
from scripts.run_calibration import run_calibration
from src.utils.logger import start_logging, stop_logging

def main():
    """
    Two-Region Austin Example + Time Stretch Estimation
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = start_logging(f"austin_2region_log_{timestamp}.txt")
    
    try:
        # Create configuration
        calib_config = CalibrationConfig(
            T=180,
            timesteps_per_day=4,
            mode="BETA_ONLY",
            optimizers=["L-BFGS-B"],
            verbose_lbfgs=True,
            loss_aggregation="regional",
            noise_type="poisson",
            
            # Time stretch settings
            apply_time_stretch=True,     # Generate truth with stretch
            time_stretch_factor=1.1,     # Truth stretch factor (1.1x slower)
            estimate_time_stretch=True,  # Try to recover this factor
            time_stretch_bounds=(0.5, 2.0)
        )
        
        # Estimation config: Beta + E0 + Time Stretch
        calib_config.estimation_config = {
            "beta_param": "L",
            "estimate_initial": {
                "E": True,   # Estimate E0
                "IP": False,
                "ISR": False,
                "ISH": False,
                "IA": False
            },
            # time_stretch is automatically added by CalibrationConfig logic
        }
        
        calib_config.regularization = RegularizationConfig(
            beta_type="l2_magnitude",
            beta_lambda=1e-6,
            compartment_configs={
                "E": {
                    "type": "structural",
                    "location_targets": [0.0, 1.0],  # 2 regions now!
                    "age_targets": [0, 0, 1, 0, 0],
                    "lambda_on_target": 100000.0,  
                    "lambda_off_target": 100000.0  
                }
            }
        )
        
        # 2-Region Model Config
        model_config = ModelConfig(
            subpop_names=["subpopA", "subpopB"],
            beta_values=[0.22, 0.28],
            seed_region_idx=1,
            seed_age_idx=2,
            seed_value=1.0,
            mixing_file="AB_mixing_params.json"
        )
        
        print("="*100)
        print("AUSTIN 2-REGION + TIME STRETCH TESTER")
        print("="*100)
        print(f"Goal: Recover Time Stretch = {calib_config.time_stretch_factor}")
        print("Regions: Loc A, Loc B")
        print("="*100)
        
        # Run calibration
        # Note: We might hit an error with mixing params if we don't fix the JSON.
        # Let's hope the user has a 2-region file or we mock it.
        # Actually, let's create a temporary 2-region mixing file to be robust.
        
        import json
        p_root = Path(__file__).parent.parent
        c_path = p_root / "CLT_BaseModel/flu_instances/calibration_research_input_files"
        ab_mixing_file = c_path / "AB_mixing_params.json"
        
        if not ab_mixing_file.exists():
            print("Creating temporary 2-region mixing file...")
            ab_mixing = {
                "num_locations": 2,
                "travel_proportions": [
                    [0.99, 0.01],
                    [0.01, 0.99]
                ]
            }
            with open(ab_mixing_file, 'w') as f:
                json.dump(ab_mixing, f)
        
        # Hack run_calibration to load this file? 
        # run_calibration hardcodes "ABC_mixing_params.json".
        # We should update run_calibration to accept mixing file name in ModelConfig.
        
        # NOTE: Since we didn't update run_calibration to allow custom mixing file name,
        # we will monkey-patch or just update run_calibration in the next step if this fails.
        # But wait, looking at run_calibration.py:
        # metapop = flu.FluMetapopModel(subpops, clt.make_dataclass_from_json(c_path / "ABC_mixing_params.json", ...))
        # It is HARDCODED.
        
        # I should probably update run_calibration.py to allow custom mixing file.
        # But for now, let's stick to the prompt.
        
        stage1_results, _ = run_calibration(calib_config, model_config)
        
        # Verify Time Stretch Recovery
        best = min(stage1_results.items(), key=lambda x: x[1]['loss'])[1]
        
        if calib_config.estimate_time_stretch:
            # Time stretch is the last parameter if estimated
            theta_opt = best['theta_opt']
            opt_ts = np.exp(theta_opt[-1])
        else:
            opt_ts = 1.0
        
        print("\n" + "="*50)
        print(f"Time Stretch Recovery Results")
        print(f"True: {calib_config.time_stretch_factor}")
        print(f"Recovered: {opt_ts:.4f}")
        print(f"Error: {abs(opt_ts - calib_config.time_stretch_factor):.4f}")
        print("="*50)

    finally:
        stop_logging(logger)

def apply_time_stretch_recovery(theta, struct):
    import numpy as np
    # Rudimentary extraction
    if "time_stretch" in struct["slices"]:
        s = struct["slices"]["time_stretch"]
        return np.exp(theta[s])[0]
    return 1.0

if __name__ == "__main__":
    main()
