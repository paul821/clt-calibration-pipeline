# Libraries to import:
import numpy as np
import torch
from typing import Dict, List, Optional
from ..utils.metrics import (
    print_beta_e0_table,
    print_multi_compartment_table,
    print_results_table,
    print_optimizer_comparison_table
)

def generate_stage1_report(
    true_params: Dict,
    best_results: Dict,
    structure: Dict,
    scale_factors: Dict,
    truth_data: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    config
):
    """
    Generate comprehensive Stage 1 report
    
    Args:
        true_params: dict with true parameter values
        best_results: dict mapping optimizer name → best result
        structure: theta structure
        scale_factors: scale factors
        truth_data: truth admissions data
        predictions_dict: dict mapping optimizer name → predictions
        config: CalibrationConfig
    """
    print("\n" + "="*100)
    print("STAGE 1 CALIBRATION REPORT")
    print("="*100)
    
    # Optimizer comparison
    print_optimizer_comparison_table(best_results, stage_name="Stage 1")
    
    # Extract parameters from best results
    opt_params_dict = {}
    for opt_name, result in best_results.items():
        theta_opt = result['theta_opt']
        opt_params = {}
        
        # Beta
        if "beta" in structure["slices"]:
            s_beta = structure["slices"]["beta"]
            opt_params["beta"] = np.exp(theta_opt[s_beta]) / scale_factors.get("beta", 1.0)
        
        # Compartments
        for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
            key = f"init_{comp_name}"
            if key in structure["slices"]:
                s_comp = structure["slices"][key]
                comp_scale = scale_factors.get(comp_name, 1.0)
                L, A, R = true_params.get(comp_name, np.zeros((3, 5, 1))).shape
                opt_params[comp_name] = (np.exp(theta_opt[s_comp]) / comp_scale).reshape(L, A, R)
        
        opt_params_dict[opt_name] = opt_params
    
    # Multi-compartment table (for best optimizer only)
    best_opt_name = min(best_results.items(), key=lambda x: x[1]['loss'])[0]
    
    compartments_to_show = ["beta"]
    for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
        if comp_name in opt_params_dict[best_opt_name]:
            compartments_to_show.append(comp_name)
    
    print_multi_compartment_table(
        true_params,
        opt_params_dict[best_opt_name],
        compartments_to_show,
        best_details=best_results[best_opt_name]
    )
    
    print("\n" + "="*100)

def generate_stage2_report(
    true_ihrs: np.ndarray,
    best_results: Dict,
    truth_data: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    config
):
    """
    Generate comprehensive Stage 2 (IHR) report
    
    Args:
        true_ihrs: true IHR values (L, A, R)
        best_results: dict mapping optimizer name → best result
        truth_data: truth admissions data (T, L, A, R)
        predictions_dict: dict mapping optimizer name → predictions
        config: CalibrationConfig
    """
    print("\n" + "="*100)
    print("STAGE 2 (IHR) CALIBRATION REPORT")
    print("="*100)
    
    # Optimizer comparison
    print_optimizer_comparison_table(best_results, stage_name="Stage 2")
    
    # Detailed IHR table for best optimizer
    best_opt_name = min(best_results.items(), key=lambda x: x[1]['loss'])[0]
    opt_ihrs = np.array(best_results[best_opt_name]['ihr_values'])
    
    print_results_table(
        f"STAGE 2 IHR RECOVERY - BEST OPTIMIZER ({best_opt_name})",
        true_ihrs,
        opt_ihrs,
        truth_data,
        predictions_dict[best_opt_name]
    )
    
    print("\n" + "="*100)

def generate_full_calibration_report(
    stage1_results: Dict,
    stage2_results: Optional[Dict],
    true_params: Dict,
    truth_data: torch.Tensor,
    config
):
    """
    Generate full 2-stage calibration report
    
    Args:
        stage1_results: Stage 1 results dict
        stage2_results: Stage 2 results dict (None if not applicable)
        true_params: true parameter values
        truth_data: truth data
        config: CalibrationConfig
    """
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*30 + "FULL CALIBRATION REPORT" + " "*45 + "#")
    print("#" + " "*98 + "#")
    print("#"*100)
    
    print(f"\nMode: {config.mode}")
    print(f"Optimizers: {', '.join(config.optimizers)}")
    print(f"Time horizon: {config.T} days")
    print(f"Loss aggregation: {config.loss_aggregation}")
    
    if config.mode == "IHR_MODE" and config.enable_gss:
        print(f"GSS enabled: Offset range {config.gss_offset_range}")
    
    # Stage 1 summary
    if stage1_results:
        print("\n" + "-"*100)
        print("STAGE 1 SUMMARY")
        print("-"*100)
        
        best_stage1 = min(stage1_results.items(), key=lambda x: x[1]['loss'])
        print(f"Best optimizer: {best_stage1[0]}")
        print(f"Final loss: {best_stage1[1]['loss']:.4f}")
        print(f"Final R²: {best_stage1[1].get('global_r2', 0.0):.6f}")
        
        if 'offset' in best_stage1[1]:
            print(f"Optimal offset: {best_stage1[1]['offset']} days")
    
    # Stage 2 summary
    if stage2_results:
        print("\n" + "-"*100)
        print("STAGE 2 SUMMARY")
        print("-"*100)
        
        best_stage2 = min(stage2_results.items(), key=lambda x: x[1]['loss'])
        print(f"Best optimizer: {best_stage2[0]}")
        print(f"Final loss: {best_stage2[1]['loss']:.4f}")
        print(f"Final R²: {best_stage2[1].get('global_r2', 0.0):.6f}")
    
    print("\n" + "#"*100)