#!/usr/bin/env python3
"""
Main calibration script
"""

# Libraries to import:
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import setup_environment
from config.calibration_config import CalibrationConfig
from config.model_config import ModelConfig
from src.optimization.stage1_gss import GSSOptimizer
from src.optimization.stage2_ihr import IHROptimizer
from src.utils.metrics import print_results_table, print_beta_e0_table
from src.visualization.plotting import save_diagnostic_plots, save_regional_aggregate_plot

# Setup environment (line 30)
setup_environment()

warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

# Now import CLT after environment setup
import clt_toolkit as clt
import flu_core as flu

def load_model_components(model_config: ModelConfig):
    """
    Load and initialize model components
    
    Migrate from lines 414-428
    """
    # Lines 414-415
    p_root = clt.utils.PROJECT_ROOT
    t_path = model_config.texas_input_path or p_root / "flu_instances/texas_input_files"
    c_path = model_config.calibration_input_path or p_root / "flu_instances/calibration_research_input_files"
    
    # Lines 416-417 (schedules)
    s_info = flu.FluSubpopSchedules(
        absolute_humidity=pd.read_csv(t_path / "absolute_humidity_austin_2023_2024.csv", index_col=0),
        flu_contact_matrix=pd.read_csv(t_path / "school_work_calendar.csv", index_col=0),
        daily_vaccines=pd.read_csv(t_path / "daily_vaccines_constant.csv", index_col=0)
    )
    
    # Lines 418 (params with time stretch)
    s_params = clt.updated_dataclass(
        clt.make_dataclass_from_json(t_path / "common_subpop_params.json", flu.FluSubpopParams),
        {attr: getattr(clt.make_dataclass_from_json(t_path / "common_subpop_params.json", flu.FluSubpopParams), attr) / 5.0 
         for attr in ['E_to_I_rate', 'IP_to_IS_rate', 'ISR_to_R_rate', 'IA_to_R_rate', 'ISH_to_H_rate', 'HR_to_R_rate', 'HD_to_D_rate', 'R_to_S_rate']}
    )
    
    # Lines 419-424 (subpopulations)
    subpops = []
    for i, (name, bt) in enumerate(zip(model_config.subpop_names, model_config.beta_values)):
        init = clt.make_dataclass_from_json(c_path / f"{name}_init_vals.json", flu.FluSubpopState)
        new_e = torch.zeros_like(torch.as_tensor(init.E)).double()
        if i == model_config.seed_region_idx:
            new_e[model_config.seed_age_idx] = model_config.seed_value
        init.E = new_e
        subpops.append(flu.FluSubpopModel(
            init, 
            clt.updated_dataclass(s_params, {"beta_baseline": bt}), 
            clt.updated_dataclass(
                clt.make_dataclass_from_json(t_path / "simulation_settings.json", flu.SimulationSettings),
                {"timesteps_per_day": model_config.timesteps_per_day}
            ),
            np.random.default_rng(),
            s_info,
            name=name
        ))
    
    # Lines 425 (metapopulation)
    metapop = flu.FluMetapopModel(
        subpops,
        clt.make_dataclass_from_json(c_path / "ABC_mixing_params.json", flu.FluMixingParams)
    )
    
    return metapop

def generate_truth_data(metapop, T, timesteps_per_day):
    """
    Generate synthetic truth data with noise
    
    Migrate from lines 426-428
    """
    d = metapop.get_flu_torch_inputs()
    with torch.no_grad():
        clean_truth_15ch = flu.torch_simulate_hospital_admits(
            d["state_tensors"], 
            d["params_tensors"], 
            d["precomputed"], 
            d["schedule_tensors"], 
            T, 
            timesteps_per_day
        )
        truth_data_15ch = torch.poisson(
            clean_truth_15ch, 
            generator=torch.Generator().manual_seed(12345)
        )
    
    return d, clean_truth_15ch, truth_data_15ch

def run_calibration(calib_config: CalibrationConfig, model_config: ModelConfig):
    """
    Main calibration workflow
    
    Orchestrates the calibration process
    """
    # Set random seeds (lines 31-32)
    torch.manual_seed(calib_config.torch_seed)
    np.random.seed(calib_config.numpy_seed)
    
    # Load model
    metapop = load_model_components(model_config)
    
    # Generate truth data
    d, clean_truth_15ch, truth_data_15ch = generate_truth_data(
        metapop, 
        calib_config.T, 
        calib_config.timesteps_per_day
    )
    
    # Baseline audit (lines 429)
    print_results_table(
        "AUDIT 1 BASELINE (POISSON NOISE LIMIT)",
        d["params_tensors"].IP_to_ISH_prop.numpy(),
        None,
        truth_data_15ch,
        clean_truth_15ch
    )
    
    # Initialize state variables (lines 430)
    cur_st, cur_pa, cur_T = d["state_tensors"], d["params_tensors"], calib_config.T
    shifted_noisy, shifted_clean = truth_data_15ch, clean_truth_15ch
    stage1_report_data = None
    
    # Stage 1: Beta & E0 optimization (lines 432-439)
    if calib_config.mode in ["BETA_ONLY", "SEQUENTIAL"]:
        gss_optimizer = GSSOptimizer(
            calib_config,
            calib_config.estimation_config,
            calib_config.reg_config,
            verbose=calib_config.verbose_lbfgs
        )
        
        best, struct, stage1_report_data = gss_optimizer.run(
            truth_data_15ch,
            clean_truth_15ch,
            d["state_tensors"],
            d["params_tensors"],
            metapop
        )
        
        # Apply optimal parameters (lines 434-439)
        from src.utils.theta_transforms import apply_gss_theta
        cur_st, cur_pa = apply_gss_theta(
            torch.from_numpy(best['theta_opt']),
            calib_config.estimation_config,
            struct,
            d["state_tensors"],
            d["params_tensors"]
        )
        cur_T, off = best['T'], best['offset']
        
        # Shift truth data (lines 437-439)
        pad_n = torch.zeros((abs(off), 3, 5, 1)) if off < 0 else None
        shifted_noisy = truth_data_15ch[off:] if off >= 0 else torch.cat([pad_n, truth_data_15ch], dim=0)[:calib_config.T]
        shifted_clean = clean_truth_15ch[off:] if off >= 0 else torch.cat([pad_n, clean_truth_15ch], dim=0)[:calib_config.T]
    
    # Stage 2: IHR optimization (lines 440-441)
    if calib_config.mode in ["IHR_ONLY", "SEQUENTIAL"]:
        ihr_optimizer = IHROptimizer(calib_config, verbose=calib_config.verbose_lbfgs)
        
        from src.utils.theta_transforms import apply_ihr_theta
        cur_pa = apply_ihr_theta(
            ihr_optimizer.run(shifted_noisy, cur_st, cur_pa, metapop, cur_T),
            cur_pa
        )
    
    # Final predictions and reporting (lines 442-449)
    with torch.no_grad():
        final_pred = flu.torch_simulate_hospital_admits(
            cur_st,
            cur_pa,
            d["precomputed"],
            d["schedule_tensors"],
            cur_T,
            calib_config.timesteps_per_day
        )
    
    print_results_table(
        f"FINAL CALIBRATION RECOVERY ({calib_config.calibration_method})",
        d["params_tensors"].IP_to_ISH_prop.numpy(),
        cur_pa.IP_to_ISH_prop.numpy().flatten(),
        shifted_noisy,
        final_pred
    )
    
    # Repeat Stage 1 report (lines 444-447)
    if stage1_report_data:
        print("\n" + "="*40 + "\nREPEAT REPORT: STAGE 1 PERFORMANCE\n" + "="*40)
        print_beta_e0_table(*stage1_report_data)
    
    # Save visualizations (lines 449-450)
    save_diagnostic_plots(shifted_noisy, shifted_clean, final_pred, cur_T)
    save_regional_aggregate_plot(shifted_noisy, shifted_clean, final_pred, cur_T)
    
    return cur_st, cur_pa, final_pred

if __name__ == "__main__":
    # Initialize configurations
    calib_config = CalibrationConfig()
    model_config = ModelConfig()
    
    # Run calibration
    final_state, final_params, predictions = run_calibration(calib_config, model_config)
    
    print("\nCalibration complete!")
