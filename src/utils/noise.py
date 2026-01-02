import torch
import numpy as np

def add_noise_to_truth(clean_truth, noise_type="none", noise_seed=12345):
    """
    Add noise to synthetic truth data for robustness testing
    
    Args:
        clean_truth: torch.Tensor of shape (T, L, A, R)
        noise_type: "none", "poisson", "gaussian"
        noise_seed: random seed for reproducibility
    
    Returns:
        noisy_truth: torch.Tensor of same shape
    """
    if noise_type == "none":
        return clean_truth.clone()
    
    elif noise_type == "poisson":
        generator = torch.Generator().manual_seed(noise_seed)
        noisy = torch.poisson(clean_truth, generator=generator)
        return noisy
    
    elif noise_type == "gaussian":
        torch.manual_seed(noise_seed)
        # Add Gaussian noise with std = sqrt(truth + epsilon)
        noise = torch.randn_like(clean_truth) * torch.sqrt(clean_truth + 1e-6)
        noisy = clean_truth + noise
        noisy = torch.clamp(noisy, min=0.0)  # Non-negative
        return noisy
    
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
