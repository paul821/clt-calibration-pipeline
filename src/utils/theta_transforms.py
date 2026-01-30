import torch
import copy
import numpy as np
from typing import Dict, Tuple

def build_gss_theta_structure(config, base_state, base_params):
    """
    Build theta structure for GSS Stage 1
    
    Returns dict with:
        - "slices": dict mapping parameter family to slice in theta
        - "size": total length of theta vector
    """
    L, A, R = base_params.beta_baseline.shape
    slices = {}
    idx = 0
    
    # Beta block
    if config["beta_param"] == "L":
        n_beta = L
    elif config["beta_param"] == "LA":
        n_beta = L * A
    else:
        raise ValueError("beta_param must be 'L' or 'LA'")
    
    slices["beta"] = slice(idx, idx + n_beta)
    idx += n_beta
    
    # Initial compartment blocks
    for comp_name, do_est in config["estimate_initial"].items():
        if not do_est:
            continue
        comp_tensor = getattr(base_state, comp_name)
        n_comp = comp_tensor.numel()  # L × A × R
        slices[f"init_{comp_name}"] = slice(idx, idx + n_comp)
        idx += n_comp
    
    # Time stretch
    if config.get("estimate_time_stretch", False):
        slices["time_stretch"] = slice(idx, idx + 1)
        idx += 1
    
    return {"slices": slices, "size": idx, "beta_param": config["beta_param"], "estimate_time_stretch": config.get("estimate_time_stretch", False)}

def apply_gss_theta(theta, config, structure, base_state, base_params, scale_factors):
    """
    Apply theta vector to create state and params
    
    Args:
        theta: torch.Tensor (log-scaled parameters)
        config: estimation config dict
        structure: theta structure from build_gss_theta_structure
        base_state: base state object
        base_params: base params object
        scale_factors: dict of scale factors per parameter family
    
    Returns:
        (init_state, params)
    """
    L, A, R = base_params.beta_baseline.shape
    slices = structure["slices"]
    
    s_beta = slices["beta"]
    theta_beta = theta[s_beta]
    # theta stores log(beta * scale)
    beta_vals = torch.exp(theta_beta) / scale_factors["beta"]
    
    if structure["beta_param"] == "L":
        beta_tensor = beta_vals.view(L, 1, 1).expand(L, A, R)
    elif structure["beta_param"] == "LA":
        beta_tensor = beta_vals.view(L, A, 1).expand(L, A, R)
    else:
        raise ValueError("beta_param must be 'L' or 'LA'")
    
    params = copy.deepcopy(base_params)
    params.beta_baseline = beta_tensor.double()
    
    init_state = copy.deepcopy(base_state)
    
    for comp_name, do_est in config["estimate_initial"].items():
        if not do_est:
            continue
        
        key = f"init_{comp_name}"
        if key not in slices:
            continue
        
        s_comp = slices[key]
        theta_comp = theta[s_comp]
        
        comp_scale = float(scale_factors.get(comp_name, 1.0))
        # theta stores log(comp * scale)
        comp_vals = (torch.exp(theta_comp) / comp_scale).view_as(getattr(init_state, comp_name))
        setattr(init_state, comp_name, comp_vals.double())
    
    # Time stretch
    if structure.get("estimate_time_stretch", False):
        s_ts = slices["time_stretch"]
        theta_ts = theta[s_ts]
        # theta stores log(time_stretch)
        # Note: we don't scale time_stretch currently as it's close to 1.0
        time_stretch = torch.exp(theta_ts)
        return init_state, params, time_stretch
    
    return init_state, params, 1.0

def apply_ihr_theta(theta, base_params, scale_factors):
    """
    Apply IHR theta to params (Stage 2)
    
    Args:
        theta: numpy array or torch.Tensor (log-scaled IHR values)
        base_params: base params object
        scale_factors: dict with "ihr" key
    
    Returns:
        params with updated IP_to_ISH_prop
    """
    params = copy.deepcopy(base_params)
    
    if isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta).to(torch.float64)
    
    # theta stores log(IHR * scale)
    ihr_vals = torch.exp(theta) / scale_factors.get("ihr", 1.0)
    
    # Reshape to (L, A, R) - assumes IHR per (location, age)
    # If IHR param is "L", expand across A and R
    L, A, R = base_params.beta_baseline.shape
    
    if ihr_vals.numel() == L:
        ihr_tensor = ihr_vals.view(L, 1, 1).expand(L, A, R)
    elif ihr_vals.numel() == L * A:
        ihr_tensor = ihr_vals.view(L, A, 1).expand(L, A, R)
    elif ihr_vals.numel() == L * A * R:
        ihr_tensor = ihr_vals.view(L, A, R)
    else:
        raise ValueError(f"IHR theta size {ihr_vals.numel()} doesn't match expected dimensions")
    
    params.IP_to_ISH_prop = ihr_tensor.double().contiguous()
    return params

def build_multi_optimizer_theta_structure(config, base_state, base_params):
    """
    Build theta structure for multi-optimizer suite
    
    Handles beta, IHR, and multiple initial compartments
    """
    L, A, R = base_params.beta_baseline.shape
    slices = {}
    idx = 0
    
    # Beta
    estimate_beta = config.get("estimate_beta", True)
    if estimate_beta:
        beta_param = config.get("beta_param", "L")
        if beta_param == "L":
            n_beta = L
        elif beta_param == "LA":
            n_beta = L * A
        else:
            raise ValueError("beta_param must be 'L' or 'LA'")
        
        slices["beta"] = slice(idx, idx + n_beta)
        idx += n_beta
    
    # IHR (if estimating)
    estimate_ihr = config.get("estimate_ihr", False)
    if estimate_ihr:
        ihr_param = config.get("ihr_param", "L")
        if ihr_param == "L":
            n_ihr = L
        elif ihr_param == "LAR":
            n_ihr = L * A * R
        else:
            raise ValueError("ihr_param must be 'L' or 'LAR'")
        
        slices["ihr"] = slice(idx, idx + n_ihr)
        idx += n_ihr
    
    # Initial compartments
    for comp_name, do_est in config.get("estimate_initial", {}).items():
        if not do_est:
            continue
        comp_tensor = getattr(base_state, comp_name)
        n_comp = comp_tensor.numel()
        slices[f"init_{comp_name}"] = slice(idx, idx + n_comp)
        idx += n_comp
    
        idx += n_comp
    
    # Time stretch
    if config.get("estimate_time_stretch", False):
        slices["time_stretch"] = slice(idx, idx + 1)
        idx += 1
    
    return {
        "slices": slices,
        "size": idx,
        "beta_param": config.get("beta_param", "L"),
        "ihr_param": config.get("ihr_param", "L") if estimate_ihr else None,
        "estimate_time_stretch": config.get("estimate_time_stretch", False)
    }
    
def apply_multi_optimizer_theta(theta, config, structure, base_state, base_params, scale_factors):
    """
    Apply theta for multi-optimizer suite
    
    """
    if isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta).to(torch.float64)
    
    L, A, R = base_params.beta_baseline.shape
    slices = structure["slices"]
    
    params = copy.deepcopy(base_params)
    init_state = copy.deepcopy(base_state)
    
    # Beta
    if "beta" in slices:
        s_beta = slices["beta"]
        theta_beta = theta[s_beta]  # Extract beta slice
        beta_vals = torch.exp(theta_beta) / scale_factors.get("beta", 1.0)
        
        beta_param = structure.get("beta_param", "L")
        
        if beta_param == "L":
            # Per-location: expect L values
            if beta_vals.numel() != L:
                raise ValueError(f"Expected {L} beta values for beta_param='L', got {beta_vals.numel()}")
            params.beta_baseline = beta_vals.view(L, 1, 1).expand(L, A, R).double()
        elif beta_param == "LA":
            # Per-location-age: expect L*A values
            if beta_vals.numel() != L * A:
                raise ValueError(f"Expected {L*A} beta values for beta_param='LA', got {beta_vals.numel()}")
            params.beta_baseline = beta_vals.view(L, A, 1).expand(L, A, R).double()
        else:
            raise ValueError(f"Unknown beta_param: {beta_param}")
    
    # IHR
    if "ihr" in slices:
        s_ihr = slices["ihr"]
        theta_ihr = theta[s_ihr]
        ihr_vals = torch.exp(theta_ihr) / scale_factors.get("ihr", 1.0)
        
        ihr_param = structure.get("ihr_param", "L")
        
        if ihr_param == "L":
            if ihr_vals.numel() != L:
                raise ValueError(f"Expected {L} IHR values for ihr_param='L', got {ihr_vals.numel()}")
            params.IP_to_ISH_prop = ihr_vals.view(L, 1, 1).expand(L, A, R).double().contiguous()
        elif ihr_param == "LAR":
            if ihr_vals.numel() != L * A * R:
                raise ValueError(f"Expected {L*A*R} IHR values for ihr_param='LAR', got {ihr_vals.numel()}")
            params.IP_to_ISH_prop = ihr_vals.view(L, A, R).double().contiguous()
        else:
            raise ValueError(f"Unknown ihr_param: {ihr_param}")
    
    # Initial compartments
    for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
        key = f"init_{comp_name}"
        if key not in slices:
            continue
        
        s_comp = slices[key]
        theta_comp = theta[s_comp]
        comp_scale = float(scale_factors.get(comp_name, 1.0))
        
        # Expected shape for compartments
        expected_size = L * A * R
        if theta_comp.numel() != expected_size:
            raise ValueError(f"Expected {expected_size} values for {comp_name}, got {theta_comp.numel()}")
        
        comp_vals = (torch.exp(theta_comp) / comp_scale).view(L, A, R)
        setattr(init_state, comp_name, comp_vals.double())
    

    # Time stretch
    if structure.get("estimate_time_stretch", False):
        s_ts = slices["time_stretch"]
        theta_ts = theta[s_ts]
        time_stretch = torch.exp(theta_ts)
        return init_state, params, time_stretch

    return init_state, params, 1.0
