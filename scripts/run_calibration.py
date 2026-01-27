#Main calibration script - Unified pipeline for all modes

import torch
import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from typing import Dict
from src.utils.logger import start_logging, stop_logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config and utilities before environment setup
from config.calibration_config import CalibrationConfig, RegularizationConfig
from config.model_config import ModelConfig

# Setup environment
from src.environment import setup_environment
setup_environment()

warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

# Import CLT modules (after environment setup)
import clt_toolkit as clt
import flu_core as flu

# Import optimization modules
from src.optimization.stage1_gss import GSSOptimizer
from src.optimization.stage2_ihr import IHROptimizer
from src.optimization.multi_optimizer_stage1 import MultiOptimizerStage1
from src.optimization.multi_optimizer_stage2 import MultiOptimizerStage2

# Import utilities
from src.utils.metrics import print_optimizer_comparison_table
from src.utils.noise import add_noise_to_truth
from src.utils.theta_transforms import apply_gss_theta, apply_ihr_theta, apply_multi_optimizer_theta

# Import visualization
from src.visualization.plotting import (
    save_diagnostic_plots,
    save_regional_aggregate_plot,
    plot_multi_optimizer_comparison,
    plot_parameter_recovery_bars,
    plot_loss_component_breakdown
)
from src.visualization.reports import (
    generate_stage1_report,
    generate_stage2_report,
    generate_full_calibration_report
)

from src.utils.simulation_utils import apply_time_stretching

def get_output_prefix(config):
    """Get output prefix for filenames"""
    return config.output_prefix if config.output_prefix else f"{config.mode}_"



def load_model_components(model_config: ModelConfig, calib_config: CalibrationConfig):
    """
    Load and initialize model components
    """
    p_root = clt.utils.PROJECT_ROOT
    t_path = model_config.texas_input_path or p_root / "flu_instances/texas_input_files"
    c_path = model_config.calibration_input_path or p_root / "flu_instances/calibration_research_input_files"
    
    # Schedules
    s_info = flu.FluSubpopSchedules(
        absolute_humidity=pd.read_csv(t_path / "absolute_humidity_austin_2023_2024.csv", index_col=0),
        flu_contact_matrix=pd.read_csv(t_path / "school_work_calendar.csv", index_col=0),
        daily_vaccines=pd.read_csv(t_path / "daily_vaccines_constant.csv", index_col=0),
        mobility_modifier=pd.read_csv(t_path / "mobility_modifier.csv", index_col=0)
    )
    
    # Params with optional time stretching
    base_subpop_params = clt.make_dataclass_from_json(
        t_path / "common_subpop_params.json", 
        flu.FluSubpopParams
    )
    
    if calib_config.apply_time_stretch:
        print(f"Applying time stretching: factor = {calib_config.time_stretch_factor}")
        s_params = apply_time_stretching(base_subpop_params, calib_config.time_stretch_factor)
    else:
        s_params = base_subpop_params
    
    # Subpopulations
    subpops = []
    for i, (name, bt) in enumerate(zip(model_config.subpop_names, model_config.beta_values)):
        init = clt.make_dataclass_from_json(c_path / f"{name}_init_vals.json", flu.FluSubpopState)
        
        # Set initial seeding
        # Set initial seeding
        new_e = np.zeros_like(init.E, dtype=float)
        if i == model_config.seed_region_idx:
            new_e[model_config.seed_age_idx] = model_config.seed_value
        init.E = new_e
        
        subpops.append(flu.FluSubpopModel(
            init, 
            clt.updated_dataclass(s_params, {"beta_baseline": bt}), 
            clt.updated_dataclass(
                clt.make_dataclass_from_json(t_path / "simulation_settings.json", flu.SimulationSettings),
                {
                    "timesteps_per_day": model_config.timesteps_per_day,
                    "use_deterministic_softplus": False,
                    "transition_type": "binom_deterministic_no_round"
                }
            ),
            np.random.default_rng(),
            s_info,
            name=name
        ))
    
    # Metapopulation
    metapop = flu.FluMetapopModel(
        subpops,
        clt.make_dataclass_from_json(c_path / model_config.mixing_file, flu.FluMixingParams)
    )
    
    return metapop

def generate_truth_data(metapop, calib_config: CalibrationConfig, model_config: ModelConfig):
    """
    Generate synthetic truth data with optional noise
    """
    d = metapop.get_flu_torch_inputs()
    
    with torch.no_grad():
        clean_truth_15ch = flu.torch_simulate_hospital_admits(
            d["state_tensors"], 
            d["params_tensors"], 
            d["precomputed"], 
            d["schedule_tensors"], 
            calib_config.T, 
            calib_config.timesteps_per_day
        )
        
        # Add noise if requested
        truth_data_15ch = add_noise_to_truth(
            clean_truth_15ch,
            noise_type=calib_config.noise_type,
            noise_seed=calib_config.noise_seed
        )
    
    return d, clean_truth_15ch, truth_data_15ch

def run_ihr_mode(
    calib_config: CalibrationConfig,
    model_config: ModelConfig,
    metapop,
    d: Dict,
    clean_truth_15ch: torch.Tensor,
    truth_data_15ch: torch.Tensor
):
    """
    Run IHR_MODE calibration
    
    Two stages:
        Stage 1: Beta + initial compartments (with optional GSS offset discovery)
        Stage 2: IHR (holding Stage 1 parameters fixed)
    """
    print("\n" + "="*100)
    print("IHR MODE CALIBRATION")
    print("="*100)
    
    prefix = get_output_prefix(calib_config)
    
    # Stage 1
    print("\n### STAGE 1: TRANSMISSION & SEEDING ###")
    
    gss_optimizer = GSSOptimizer(
        calib_config,
        calib_config.estimation_config,
        model_config.scale_factors,
        verbose=calib_config.verbose_lbfgs
    )
    
    stage1_results, struct, all_stage1_attempts = gss_optimizer.run(
        truth_data_15ch,
        clean_truth_15ch,
        d["state_tensors"],
        d["params_tensors"],
        metapop
    )
    
    # Print Stage 1 comparison
    print_optimizer_comparison_table(stage1_results, stage_name="Stage 1")
    
    # Plot Stage 1 optimizer comparison
    plot_multi_optimizer_comparison(
        stage1_results,
        stage_name="Stage 1",
        save_path="stage1_optimizer_comparison.png",
        prefix=prefix
    )
    
    # Plot loss breakdown
    plot_loss_component_breakdown(
        stage1_results,
        stage_name="Stage 1",
        save_path="stage1_loss_breakdown.png",
        prefix=prefix
    )
    
    # Select best Stage 1 result (minimum loss across all optimizers)
    best_stage1 = min(stage1_results.items(), key=lambda x: x[1]['loss'])
    best_opt_name_s1 = best_stage1[0]
    best_result_s1 = best_stage1[1]
    
    print(f"\n>>> BEST STAGE 1 OPTIMIZER: {best_opt_name_s1} <<<")
    print(f"Loss: {best_result_s1['loss']:.4f}, R²: {best_result_s1['global_r2']:.6f}")
    
    # Apply best Stage 1 parameters
    best_offset = best_result_s1['offset']
    best_T = best_result_s1['T']
    
    final_state_s1, final_params_s1 = apply_gss_theta(
        torch.from_numpy(best_result_s1['theta_opt']),
        calib_config.estimation_config,
        struct,
        d["state_tensors"],
        d["params_tensors"],
        model_config.scale_factors
    )
    
    # Shift truth data according to best offset
    if best_offset >= 0:
        shifted_truth = truth_data_15ch[best_offset:]
        shifted_clean = clean_truth_15ch[best_offset:]
    else:
        pad_n = torch.zeros((abs(best_offset), 3, 5, 1))
        shifted_truth = torch.cat([pad_n, truth_data_15ch], dim=0)[:calib_config.T]
        shifted_clean = torch.cat([pad_n, clean_truth_15ch], dim=0)[:calib_config.T]
    
    # Generate Stage 1 predictions for all optimizers
    stage1_predictions = {}
    for opt_name, result in stage1_results.items():
        opt_offset = result['offset']
        opt_T = result['T']
        opt_state, opt_params = apply_gss_theta(
            torch.from_numpy(result['theta_opt']),
            calib_config.estimation_config,
            struct,
            d["state_tensors"],
            d["params_tensors"],
            model_config.scale_factors
        )
        
        # Shift for this optimizer's offset
        if opt_offset >= 0:
            opt_shifted_truth = truth_data_15ch[opt_offset:]
        else:
            pad_n = torch.zeros((abs(opt_offset), 3, 5, 1))
            opt_shifted_truth = torch.cat([pad_n, truth_data_15ch], dim=0)[:calib_config.T]
        
        with torch.no_grad():
            inputs = metapop.get_flu_torch_inputs()
            stage1_predictions[opt_name] = flu.torch_simulate_hospital_admits(
                opt_state, opt_params, inputs["precomputed"], inputs["schedule_tensors"],
                opt_T, calib_config.timesteps_per_day
            )
    
    # Extract true parameters for reporting
    true_params_s1 = {
        "beta": d["params_tensors"].beta_baseline[:, 0, 0].detach().cpu().numpy()
    }
    
    for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
        if calib_config.estimation_config["estimate_initial"].get(comp_name, False):
            true_params_s1[comp_name] = getattr(d["state_tensors"], comp_name).detach().cpu().numpy()
    
    # Generate Stage 1 report
    generate_stage1_report(
        true_params_s1,
        stage1_results,
        struct,
        model_config.scale_factors,
        shifted_truth,
        stage1_predictions,
        calib_config
    )
    
    # Plot parameter recovery
    opt_params_s1 = {}
    for opt_name, result in stage1_results.items():
        theta_opt = result['theta_opt']
        opt_params = {}
        
        if "beta" in struct["slices"]:
            s_beta = struct["slices"]["beta"]
            opt_params["beta"] = np.exp(theta_opt[s_beta]) / model_config.scale_factors.get("beta", 1.0)
        
        for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
            key = f"init_{comp_name}"
            if key in struct["slices"]:
                s_comp = struct["slices"][key]
                comp_scale = model_config.scale_factors.get(comp_name, 1.0)
                L, A, R = true_params_s1.get(comp_name, np.zeros((3, 5, 1))).shape
                opt_params[comp_name] = (np.exp(theta_opt[s_comp]) / comp_scale).reshape(L, A, R)
        
        opt_params_s1[opt_name] = opt_params
    
    plot_parameter_recovery_bars(
        true_params_s1,
        opt_params_s1,
        param_names=["beta"] + [c for c in ["E", "IP", "ISR", "ISH", "IA"] if c in true_params_s1],
        save_path="stage1_parameter_recovery.png",
        prefix=prefix
    )
    
    # Stage 2
    print("\n### STAGE 2: INFECTION-HOSPITALIZATION RATE ###")
    
    ihr_optimizer = IHROptimizer(
        calib_config,
        model_config.scale_factors,
        verbose=calib_config.verbose_lbfgs
    )
    
    stage2_results, all_stage2_attempts = ihr_optimizer.run(
        shifted_truth,
        final_state_s1,
        final_params_s1,
        metapop,
        best_T
    )
    
    # Print Stage 2 comparison
    print_optimizer_comparison_table(stage2_results, stage_name="Stage 2")
    
    # Plot Stage 2 optimizer comparison
    plot_multi_optimizer_comparison(
        stage2_results,
        stage_name="Stage 2",
        save_path="stage2_optimizer_comparison.png",
        prefix=prefix
    )
    
    # Generate Stage 2 predictions
    stage2_predictions = {}
    for opt_name, result in stage2_results.items():
        opt_params = apply_ihr_theta(
            torch.from_numpy(result['theta_opt']),
            final_params_s1,
            model_config.scale_factors
        )
        
        # Update metapop
        metapop._full_metapop_params_tensors.IP_to_ISH_prop = opt_params.IP_to_ISH_prop.detach()
        for i, sub in enumerate(metapop.subpop_models.values()):
            sub.params = clt.updated_dataclass(
                sub.params,
                {"IP_to_ISH_prop": opt_params.IP_to_ISH_prop.detach()[i]}
            )
        
        with torch.no_grad():
            inputs = metapop.get_flu_torch_inputs()
            stage2_predictions[opt_name] = flu.torch_simulate_hospital_admits(
                final_state_s1, opt_params, inputs["precomputed"], inputs["schedule_tensors"],
                best_T, calib_config.timesteps_per_day
            )
    
    # Generate Stage 2 report
    true_ihrs = d["params_tensors"].IP_to_ISH_prop.detach().cpu().numpy()
    generate_stage2_report(
        true_ihrs,
        stage2_results,
        shifted_truth,
        stage2_predictions,
        calib_config
    )
    
    # Final diagnostic plots
    best_stage2 = min(stage2_results.items(), key=lambda x: x[1]['loss'])
    best_opt_name_s2 = best_stage2[0]
    
    save_diagnostic_plots(
        shifted_truth,
        shifted_clean,
        stage2_predictions[best_opt_name_s2],
        best_T,
        age_labels=calib_config.age_labels,
        prefix=prefix
    )
    
    save_regional_aggregate_plot(
        shifted_truth,
        shifted_clean,
        stage2_predictions[best_opt_name_s2],
        best_T,
        filename="final_calibration_regional_aggregate.png",
        prefix=prefix
    )
    
    # Full calibration report
    generate_full_calibration_report(
        stage1_results,
        stage2_results,
        true_params_s1,
        shifted_truth,
        calib_config
    )
    
    return stage1_results, stage2_results

def run_multi_optimizer_mode(
    calib_config: CalibrationConfig,
    model_config: ModelConfig,
    metapop,
    d: Dict,
    clean_truth_15ch: torch.Tensor,
    truth_data_15ch: torch.Tensor
):
    """
    Run multi-optimizer calibration 
    
    Used for modes: BETA_ONLY, SEQUENTIAL
    """
    print("\n" + "="*100)
    print(f"{calib_config.mode} CALIBRATION")
    print("="*100)
    
    prefix = get_output_prefix(calib_config)
    
    # Stage 1
    print("\n### STAGE 1: TRANSMISSION & SEEDING ###")
    
    multi_opt_s1 = MultiOptimizerStage1(
        calib_config,
        calib_config.estimation_config,
        model_config.scale_factors
    )
    
    results_df_s1, best_per_opt_s1, struct = multi_opt_s1.run(
        truth_data_15ch,
        clean_truth_15ch,
        d["state_tensors"],
        d["params_tensors"],
        metapop
    )
    
    # Print comparison
    print_optimizer_comparison_table(best_per_opt_s1, stage_name="Stage 1")
    
    # Plot comparisons
    plot_multi_optimizer_comparison(
        best_per_opt_s1,
        stage_name="Stage 1",
        save_path="stage1_optimizer_comparison.png",
        prefix=prefix
    )
    
    # Generate predictions
    stage1_predictions = {}
    for opt_name, result in best_per_opt_s1.items():
        opt_state, opt_params, opt_ts = apply_multi_optimizer_theta(
            torch.from_numpy(result['theta_opt']),
            calib_config.estimation_config,
            struct,
            d["state_tensors"],
            d["params_tensors"],
            model_config.scale_factors
        )
        
        if calib_config.estimate_time_stretch:
            opt_params = apply_time_stretching(opt_params, opt_ts)
        elif calib_config.apply_time_stretch:
            opt_params = apply_time_stretching(opt_params, calib_config.time_stretch_factor)
        
        with torch.no_grad():
            inputs = metapop.get_flu_torch_inputs()
            stage1_predictions[opt_name] = flu.torch_simulate_hospital_admits(
                opt_state, opt_params, inputs["precomputed"], inputs["schedule_tensors"],
                calib_config.T, calib_config.timesteps_per_day
            )
    
    # Extract true parameters
    true_params_s1 = {
        "beta": d["params_tensors"].beta_baseline[:, 0, 0].detach().cpu().numpy()
    }
    
    for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
        if calib_config.estimation_config["estimate_initial"].get(comp_name, False):
            true_params_s1[comp_name] = getattr(d["state_tensors"], comp_name).detach().cpu().numpy()
    
    # Generate report
    generate_stage1_report(
        true_params_s1,
        best_per_opt_s1,
        struct,
        model_config.scale_factors,
        truth_data_15ch,
        stage1_predictions,
        calib_config
    )
    
    # Diagnostic plots
    best_s1 = min(best_per_opt_s1.items(), key=lambda x: x[1]['loss'])
    best_opt_name_s1 = best_s1[0]
    
    save_diagnostic_plots(
        truth_data_15ch,
        clean_truth_15ch,
        stage1_predictions[best_opt_name_s1],
        calib_config.T,
        age_labels=calib_config.age_labels,
        prefix=prefix
    )
    
    save_regional_aggregate_plot(
        truth_data_15ch,
        clean_truth_15ch,
        stage1_predictions[best_opt_name_s1],
        calib_config.T,
        filename="stage1_calibration_regional_aggregate.png",
        prefix=prefix
    )
    
    # Stage 2
    if calib_config.mode == "SEQUENTIAL":
        print("\n### STAGE 2: INFECTION-HOSPITALIZATION RATE ###")
        
        # Use best Stage 1 result as starting point
        best_state_s1, best_params_s1 = apply_multi_optimizer_theta(
            torch.from_numpy(best_s1[1]['theta_opt']),
            calib_config.estimation_config,
            struct,
            d["state_tensors"],
            d["params_tensors"],
            model_config.scale_factors
        )
        
        multi_opt_s2 = MultiOptimizerStage2(
            calib_config,
            model_config.scale_factors
        )
        
        results_df_s2, best_per_opt_s2 = multi_opt_s2.run(
            truth_data_15ch,
            best_state_s1,
            best_params_s1,
            metapop,
            calib_config.T
        )
        
        # Print comparison
        print_optimizer_comparison_table(best_per_opt_s2, stage_name="Stage 2")
        
        # Plot comparison
        plot_multi_optimizer_comparison(
            best_per_opt_s2,
            stage_name="Stage 2",
            save_path="stage2_optimizer_comparison.png",
            prefix=prefix
        )
        
        # Generate predictions
        stage2_predictions = {}
        for opt_name, result in best_per_opt_s2.items():
            opt_params = apply_ihr_theta(
                torch.from_numpy(result['theta_opt']),
                best_params_s1,
                model_config.scale_factors
            )
            
            with torch.no_grad():
                inputs = metapop.get_flu_torch_inputs()
                stage2_predictions[opt_name] = flu.torch_simulate_hospital_admits(
                    best_state_s1, opt_params, inputs["precomputed"], inputs["schedule_tensors"],
                    calib_config.T, calib_config.timesteps_per_day
                )
        
        # Generate report
        true_ihrs = d["params_tensors"].IP_to_ISH_prop.detach().cpu().numpy()
        generate_stage2_report(
            true_ihrs,
            best_per_opt_s2,
            truth_data_15ch,
            stage2_predictions,
            calib_config
        )
        
        # Final diagnostic plots
        best_s2 = min(best_per_opt_s2.items(), key=lambda x: x[1]['loss'])
        best_opt_name_s2 = best_s2[0]
        
        save_diagnostic_plots(
            truth_data_15ch,
            clean_truth_15ch,
            stage2_predictions[best_opt_name_s2],
            calib_config.T,
            age_labels=calib_config.age_labels,
            prefix=prefix
        )
        
        save_regional_aggregate_plot(
            truth_data_15ch,
            clean_truth_15ch,
            stage2_predictions[best_opt_name_s2],
            calib_config.T,
            filename="final_calibration_regional_aggregate.png",
            prefix=prefix
        )
        
        # Full report
        generate_full_calibration_report(
            best_per_opt_s1,
            best_per_opt_s2,
            true_params_s1,
            truth_data_15ch,
            calib_config
        )
        
        return best_per_opt_s1, best_per_opt_s2
    
    return best_per_opt_s1, None

def run_calibration(calib_config: CalibrationConfig, model_config: ModelConfig):
    """
    Main calibration workflow - routes to appropriate mode
    """
    # Set random seeds
    torch.manual_seed(calib_config.torch_seed)
    np.random.seed(calib_config.numpy_seed)
    
    # Load model
    metapop = load_model_components(model_config, calib_config)
    
    # Generate truth data
    d, clean_truth_15ch, truth_data_15ch = generate_truth_data(metapop, calib_config, model_config)
    
    # Print baseline audit
    from src.utils.metrics import print_results_table
    print_results_table(
        f"BASELINE AUDIT ({calib_config.noise_type.upper()} NOISE)",
        d["params_tensors"].IP_to_ISH_prop.numpy(),
        None,
        truth_data_15ch,
        clean_truth_15ch
    )
    
    # Route to appropriate calibration mode
    if calib_config.mode == "IHR_MODE":
        return run_ihr_mode(
            calib_config, model_config, metapop,
            d, clean_truth_15ch, truth_data_15ch
        )
    else:
        return run_multi_optimizer_mode(
            calib_config, model_config, metapop,
            d, clean_truth_15ch, truth_data_15ch
        )

if __name__ == "__main__":
    # Start logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = start_logging(f"calibration_log_{timestamp}.txt")
    
    try:
        # Initialize configurations
        calib_config = CalibrationConfig()
        model_config = ModelConfig()
        
        # Run calibration
        stage1_results, stage2_results = run_calibration(calib_config, model_config)
        
        print("\n" + "="*100)
        print("CALIBRATION COMPLETE!")
        print("="*100)
    
    finally:
        # Stop logging and restore console
        stop_logging(logger)
