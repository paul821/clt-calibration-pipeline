import torch
import numpy as np
from typing import Optional, Dict, List
from .base_loss import LossFunction, LossComponents

class RegionalLossFunction(LossFunction):
    """
    Regional loss decomposition
    
    Computes SSE per location, then sums. Preserves regional structure
    for better identifiability and interpretability.
    """
    
    def __init__(self, config, aggregation_mode="regional", timesteps_per_day=4):
        """
        aggregation_mode:
            "regional": sum over (T, A, R) per location → 3 SSE terms
            "location_age": sum over (T, R) per (location, age) → 15 SSE terms
            "global": single global SSE (legacy)
        """
        super().__init__(config, timesteps_per_day)
        self.aggregation_mode = aggregation_mode
    
    def compute_sse_decomposition(self, pred, obs):
        """
        Returns:
            regional_sse: list of SSE values per region
            total_sse: sum of regional_sse
        """
        if self.aggregation_mode == "regional":
            # Aggregate over age and risk: (T, L, A, R) → (T, L)
            pred_regional = pred.sum(dim=(-2, -1))  # sum over A, R
            obs_regional = obs.sum(dim=(-2, -1))
            
            regional_sse = [
                torch.sum((pred_regional[:, i] - obs_regional[:, i])**2).item()
                for i in range(pred_regional.shape[1])
            ]
            
        elif self.aggregation_mode == "location_age":
            # Per (location, age): (T, L, A, R) → (T, L, A)
            pred_la = pred.sum(dim=-1)  # sum over R
            obs_la = obs.sum(dim=-1)
            
            L, A = pred_la.shape[1], pred_la.shape[2]
            regional_sse = [
                torch.sum((pred_la[:, loc, age] - obs_la[:, loc, age])**2).item()
                for loc in range(L) for age in range(A)
            ]
            
        elif self.aggregation_mode == "global":
            # Single global SSE
            total_sse = torch.sum((pred - obs)**2).item()
            regional_sse = [total_sse]
            
        else:
            raise ValueError(f"Unknown aggregation_mode: {self.aggregation_mode}")
        
        total_sse = sum(regional_sse)
        return regional_sse, total_sse
    
    def compute_r2(self, pred, obs):
        """
        Compute global and regional R² values
        """
        # Global R2
        obs_flat = obs.sum(dim=(1, 2, 3))
        pred_flat = pred.sum(dim=(1, 2, 3))
        ss_tot_global = torch.sum((obs_flat - torch.mean(obs_flat))**2).item()
        ss_res_global = torch.sum((obs_flat - pred_flat)**2).item()
        global_r2 = 1.0 - (ss_res_global / ss_tot_global) if ss_tot_global > 0 else 0.0
        
        # Regional R2 (per location)
        obs_regional = obs.sum(dim=(-2, -1))  # (T, L)
        pred_regional = pred.sum(dim=(-2, -1))
        
        regional_r2 = []
        for i in range(obs_regional.shape[1]):
            obs_loc = obs_regional[:, i]
            pred_loc = pred_regional[:, i]
            ss_tot = torch.sum((obs_loc - torch.mean(obs_loc))**2).item()
            ss_res = torch.sum((obs_loc - pred_loc)**2).item()
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            regional_r2.append(r2)
        
        return global_r2, regional_r2
    
    def __call__(self, pred, obs, regularization_terms: Optional[Dict[str, torch.Tensor]] = None):
        """
        Compute regional loss with optional regularization
        
        Returns:
            LossComponents object
        """
        regional_sse, total_sse = self.compute_sse_decomposition(pred, obs)
        global_r2, regional_r2 = self.compute_r2(pred, obs)
        
        # Add regularization if provided
        reg_dict = {}
        total_reg = 0.0
        if regularization_terms:
            for name, term in regularization_terms.items():
                val = term.item() if torch.is_tensor(term) else float(term)
                reg_dict[name] = val
                total_reg += val
        
        total_loss = total_sse + total_reg
        
        return LossComponents(
            total_loss=total_loss,
            sse=total_sse,
            regularization=reg_dict,
            global_r2=global_r2,
            regional_r2=regional_r2,
            regional_sse=regional_sse
        )
