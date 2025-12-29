# Libraries to import:
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import numpy as np

@dataclass
class LossComponents:
    """Container for loss function components"""
    total_loss: float
    sse: float
    regularization: Dict[str, float]  # e.g., {"beta_l2": 0.5, "E_structural": 2.3}
    global_r2: float
    regional_r2: List[float]
    regional_sse: List[float]
    
    def __str__(self):
        lines = [
            f"\nLoss Breakdown:",
            f"  SSE (data fit):        {self.sse:.4f}",
        ]
        for reg_name, reg_val in self.regularization.items():
            lines.append(f"  {reg_name:20s}: {reg_val:.4f}")
        lines.extend([
            f"  {'─' * 40}",
            f"  TOTAL:                 {self.total_loss:.4f}",
            f"\nFit Quality:",
            f"  Global R²:             {self.global_r2:.6f}"
        ])
        for i, r2 in enumerate(self.regional_r2):
            lines.append(f"  Location {i} R²:       {r2:.6f}")
        return "\n".join(lines)

class LossFunction:
    """Base class for loss functions"""
    
    def __init__(self, config, timesteps_per_day=4):
        self.config = config
        self.timesteps_per_day = timesteps_per_day
        self.iteration_count = 0
        
    def __call__(self, pred, obs, weights=None):
        """Compute loss and return components"""
        raise NotImplementedError
    
    def report(self, components: LossComponents, verbosity: int = 1):
        """Print loss breakdown based on verbosity level"""
        if verbosity == 0:
            return
        elif verbosity == 1:
            print(f"Iter {self.iteration_count:03d} | Loss: {components.total_loss:.4f}, R²: {components.global_r2:.4f}")
        elif verbosity >= 2:
            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration_count}")
            print(components)
            print(f"{'='*60}")
        
        self.iteration_count += 1