# Libraries to import:
import torch
import numpy as np
from scipy.optimize import minimize
import copy

from ..utils.theta_transforms import apply_ihr_theta
from ..utils.metrics import format_iter_report
from ..loss.regional_loss import RegionalLossFunction

class IHROptimizer:
    """
    IHR optimizer for Stage 2 (Professor's approach with multi-optimizer support)
    
    ENHANCEMENTS:
    - Supports multiple optimizers (L-BFGS-B, CG, Adam, least_squares_fd)
    - Location-age loss aggregation (15-channel)
    """
    
    def __init__(self, config, scale_factors, verbose=False):
        self.config = config
        self.scale_factors = scale_factors
        self.verbose = verbose
        self.timesteps_per_day = config.timesteps_per_day
        self.optimizers = config.optimizers if hasattr(config, 'optimizers') else ["L-BFGS-B"]
    
    def run(self, truth_15ch, state_t0, params_in, metapop, current_T):
        """
        Run IHR optimization with multiple optimizers
        
        Returns:
            best_results: dict with keys per optimizer
            all_attempts: list of all optimization attempts
        """
        import flu_core as flu
        import clt_toolkit as clt
        
        x0 = params_in.IP_to_ISH_prop.detach().numpy().flatten()
        
        # Determine IHR dimension
        ihr_param = self.config.estimation_config.get("ihr_param", "L")
        L, A, R = params_in.beta_baseline.shape
        
        if ihr_param == "L":
            x0 = x0.reshape(L, A, R).mean(axis=(1, 2))  # Average to per-location
        elif ihr_param == "LAR":
            x0 = x0  # Use full (L*A*R)
        
        # Convert to log-space
        x0_log = np.log(x0 * self.scale_factors.get("ihr", 1.0))
        
        loss_fn_obj = RegionalLossFunction(
            self.config, 
            aggregation_mode="location_age",  # 15-channel
            timesteps_per_day=self.timesteps_per_day
        )
        
        results_per_optimizer = {}
        all_attempts = []
        
        for opt_name in self.optimizers:
            print(f"\n{'='*60}")
            print(f"IHR STAGE 2 - OPTIMIZER: {opt_name}")
            print(f"{'='*60}")
            
            iter_tracker = [0]
            
            def loss_fn(x_np):
                """
                Loss function returning (loss, grad)
                
                CRITICAL FIX: We need to compute gradients w.r.t. theta (IHR params)
                but metapop.get_flu_torch_inputs() doesn't like gradient-enabled tensors.
                
                Solution: 
                1. Forward pass with gradient-enabled theta to compute loss
                2. Backward to get gradients
                3. Return detached loss value and gradients
                """
                theta = torch.from_numpy(x_np).to(torch.float64).requires_grad_(True)
                
                # Apply theta to get IHR parameters
                par = apply_ihr_theta(theta, params_in, self.scale_factors)
                
                # Update metapop with DETACHED parameters for simulation
                # (metapop internals don't like gradient-enabled tensors)
                metapop._full_metapop_params_tensors.IP_to_ISH_prop = par.IP_to_ISH_prop.detach()
                for i, sub in enumerate(metapop.subpop_models.values()):
                    sub.params = clt.updated_dataclass(
                        sub.params,
                        {"IP_to_ISH_prop": par.IP_to_ISH_prop.detach()[i]}
                    )
                
                # Get inputs (this will work now with detached params)
                inputs = metapop.get_flu_torch_inputs()
                
                # But for the actual simulation that needs gradients, use par directly
                # We'll simulate with gradient-enabled params
                pred = flu.torch_simulate_hospital_admits(
                    state_t0, 
                    par,  # Use gradient-enabled params here
                    inputs["precomputed"], 
                    inputs["schedule_tensors"],
                    current_T, 
                    self.timesteps_per_day
                )
                
                # Compute MSE loss
                loss_tensor = torch.sum((pred - truth_15ch)**2)
                
                # Backward pass
                loss_tensor.backward()
                
                # Extract gradient
                grad = theta.grad.detach().numpy().copy() if theta.grad is not None else np.zeros_like(x_np)
                
                # Compute R² for reporting (detached)
                with torch.no_grad():
                    loss_components = loss_fn_obj(pred.detach(), truth_15ch)
                
                if self.verbose:
                    print(f"Iter {iter_tracker[0]:03d} | Loss: {loss_tensor.item():.4f}, "
                          f"R²: {loss_components.global_r2:.4f}")
                
                iter_tracker[0] += 1
                
                return loss_tensor.item(), grad
            
            # Run optimizer
            if opt_name == "L-BFGS-B":
                opts = {'gtol': 1e-04, 'ftol': 1e-07}
                if self.verbose:
                    opts['iprint'] = 1
                
                res = minimize(
                    lambda x: loss_fn(x)[0],
                    x0_log,
                    jac=lambda x: loss_fn(x)[1],
                    method='L-BFGS-B',
                    bounds=[(np.log(0.001 * self.scale_factors.get("ihr", 1.0)), 
                            np.log(0.50 * self.scale_factors.get("ihr", 1.0)))] * len(x0_log),
                    options=opts
                )
            
            elif opt_name == "CG":
                opts = {'gtol': 1e-04}
                if self.verbose:
                    opts['disp'] = True
                
                res = minimize(
                    lambda x: loss_fn(x)[0],
                    x0_log,
                    jac=lambda x: loss_fn(x)[1],
                    method='CG',
                    options=opts
                )
            
            elif opt_name == "Adam":
                theta = torch.tensor(x0_log, dtype=torch.float64, requires_grad=True)
                adam_optimizer = torch.optim.Adam([theta], lr=0.01)
                
                final_loss = 1e12
                for i in range(1000):
                    adam_optimizer.zero_grad()
                    loss_val, grad_np = loss_fn(theta.detach().numpy())
                    if not np.isfinite(loss_val):
                        break
                    theta.grad = torch.tensor(grad_np)
                    adam_optimizer.step()
                    final_loss = loss_val
                
                res = type('Result', (), {
                    'x': theta.detach().numpy(),
                    'fun': final_loss,
                    'success': True,
                    'message': 'Adam completed',
                    'nit': 1000
                })()
            
            elif opt_name == "least_squares_fd":
                from scipy.optimize import least_squares
                
                def residuals(x_np):
                    loss_val, _ = loss_fn(x_np)
                    return np.array([np.sqrt(max(loss_val, 0.0))])
                
                res = least_squares(residuals, x0_log, jac='2-point', max_nfev=1000)
                res.fun = res.cost * 2
            
            print(f"{opt_name} Termination: {res.message}")
            
            # Evaluate final
            with torch.no_grad():
                final_params = apply_ihr_theta(torch.from_numpy(res.x), params_in, self.scale_factors)
                metapop._full_metapop_params_tensors.IP_to_ISH_prop = final_params.IP_to_ISH_prop.detach()
                for i, sub in enumerate(metapop.subpop_models.values()):
                    sub.params = clt.updated_dataclass(
                        sub.params,
                        {"IP_to_ISH_prop": final_params.IP_to_ISH_prop.detach()[i]}
                    )
                
                inputs = metapop.get_flu_torch_inputs()
                final_pred = flu.torch_simulate_hospital_admits(
                    state_t0, final_params, inputs["precomputed"], inputs["schedule_tensors"],
                    current_T, self.timesteps_per_day
                )
                
                final_components = loss_fn_obj(final_pred, truth_15ch)
            
            result = {
                'optimizer': opt_name,
                'loss': res.fun,
                'theta_opt': res.x,
                'ihr_values': (np.exp(res.x) / self.scale_factors.get("ihr", 1.0)).tolist(),
                'global_r2': final_components.global_r2,
                'regional_r2': final_components.regional_r2,
                'regional_sse': final_components.regional_sse,
                'nit': getattr(res, 'nit', 0)
            }
            
            results_per_optimizer[opt_name] = result
            all_attempts.append(result)
        
        return results_per_optimizer, all_attempts