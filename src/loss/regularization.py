# Libraries to import:
import torch
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

class RegularizationTerm(ABC):
    """Base class for regularization terms"""
    
    @abstractmethod
    def compute(self, theta_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute regularization penalty"""
        pass

class L2MagnitudeRegularization(RegularizationTerm):
    """L2 penalty on parameter magnitude"""
    
    def __init__(self, param_name: str, lambda_val: float):
        self.param_name = param_name
        self.lambda_val = lambda_val
    
    def compute(self, theta_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.param_name not in theta_dict:
            return torch.tensor(0.0, dtype=torch.float64)
        
        param = theta_dict[self.param_name]
        # CRITICAL FIX: L2 on log-space values, not natural-space
        # (theta_dict contains natural-space values, we need log-space for proper L2)
        # Actually, no - we want L2 on natural space to penalize large natural values
        penalty = self.lambda_val * torch.sum(param ** 2)
        
        # DEBUG: Print penalty magnitude
        if penalty.item() > 0:
            print(f"L2 penalty for {self.param_name}: {penalty.item():.6e} (lambda={self.lambda_val})")
        
        return penalty

class StructuralRegularization(RegularizationTerm):
    """
    Structural regularization for initial compartments (Professor's approach)
    
    Enforces sparsity patterns: penalizes deviations from target values
    at specific (location, age) indices.
    """
    
    def __init__(
        self,
        compartment_name: str,
        location_targets: List[float],  # Length L
        age_targets: List[float],        # Length A (one-hot or continuous)
        lambda_on_target: float,
        lambda_off_target: float,
        shape: tuple  # (L, A, R)
    ):
        self.compartment_name = compartment_name
        self.location_targets = np.array(location_targets, dtype=float)
        self.age_targets = np.array(age_targets, dtype=float)
        self.lambda_on = lambda_on_target
        self.lambda_off = lambda_off_target
        self.L, self.A, self.R = shape
        
        # Build target tensor: (L, A, R)
        self.target_tensor = torch.zeros((self.L, self.A, self.R), dtype=torch.float64)
        
        # For each location and age, set target
        for loc in range(self.L):
            for age in range(self.A):
                # Target value = location_target * age_target
                target_val = self.location_targets[loc] * self.age_targets[age]
                self.target_tensor[loc, age, :] = target_val
        
        # Build weight tensor: lambda_on where target > 0, lambda_off elsewhere
        self.weight_tensor = torch.where(
            self.target_tensor > 0,
            torch.tensor(self.lambda_on, dtype=torch.float64),
            torch.tensor(self.lambda_off, dtype=torch.float64)
        )
    
    def compute(self, theta_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.compartment_name not in theta_dict:
            return torch.tensor(0.0, dtype=torch.float64)
        
        # theta_dict[compartment_name] should be (L, A, R) in natural scale
        comp_vals = theta_dict[self.compartment_name]
        
        # Ensure shape matches
        if comp_vals.shape != self.target_tensor.shape:
            comp_vals = comp_vals.view(self.L, self.A, self.R)
        
        # Weighted squared error
        penalty = torch.sum(
            self.weight_tensor * (comp_vals - self.target_tensor)**2
        )
        
        # CRITICAL FIX: Add debug logging
        if penalty.item() > 1e-10:
            print(f"Structural penalty for {self.compartment_name}: {penalty.item():.6e}")
            print(f"  On-target lambda: {self.lambda_on}, Off-target lambda: {self.lambda_off}")
            print(f"  Target sum: {self.target_tensor.sum().item():.3f}, Actual sum: {comp_vals.sum().item():.3f}")
            # Print where violations are largest
            max_violation_idx = torch.argmax(torch.abs(comp_vals - self.target_tensor))
            max_violation_val = (comp_vals.flatten()[max_violation_idx] - self.target_tensor.flatten()[max_violation_idx]).item()
            print(f"  Max violation: {max_violation_val:.6e} at index {max_violation_idx}")
        
        return penalty

def build_regularization_terms(
    config,
    shape_dict: Dict[str, tuple]  # e.g., {"beta": (L,), "E": (L, A, R)}
) -> Dict[str, RegularizationTerm]:
    """
    Build all regularization terms from config
    
    Args:
        config: CalibrationConfig with regularization settings
        shape_dict: Dictionary mapping parameter names to their shapes
    
    Returns:
        Dictionary mapping regularization names to RegularizationTerm objects
    """
    reg_config = config.regularization
    terms = {}
    
    # Beta regularization
    if reg_config.beta_type == "l2_magnitude" and "beta" in shape_dict:
        terms["beta_l2"] = L2MagnitudeRegularization("beta", reg_config.beta_lambda)
    
    # Compartment regularization
    for comp_name, comp_config in reg_config.compartment_configs.items():
        if comp_name not in shape_dict:
            continue
        
        if comp_config["type"] == "l2_magnitude":
            terms[f"{comp_name}_l2"] = L2MagnitudeRegularization(
                comp_name,
                comp_config["lambda"]
            )
        
        elif comp_config["type"] == "structural":
            L, A, R = shape_dict[comp_name]
            terms[f"{comp_name}_structural"] = StructuralRegularization(
                compartment_name=comp_name,
                location_targets=comp_config["location_targets"],
                age_targets=comp_config["age_targets"],
                lambda_on_target=comp_config["lambda_on_target"],
                lambda_off_target=comp_config["lambda_off_target"],
                shape=(L, A, R)
            )
    
    return terms