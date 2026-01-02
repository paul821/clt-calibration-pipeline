import torch
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import copy
import time as global_time

from ..utils.theta_transforms import build_gss_theta_structure, apply_gss_theta
from ..utils.metrics import format_iter_report
from ..visualization.plotting import save_convergence_plot, save_regional_aggregate_plot
from ..loss.regional_loss import RegionalLossFunction
from ..loss.regularization import build_regularization_terms

class GSSOptimizer:
    """
    Golden Section Search optimizer for Stage 1 
    
    - Supports multiple initial compartments 
    - Supports multiple optimizers (L-BFGS-B, CG, Adam, least_squares_fd)
    - Regional loss decomposition with structured regularization
    """
    
    def __init__(self, config, estimation_config, scale_factors, verbose=False):
        self.config = config
        self.estimation_config = estimation_config
        self.scale_factors = scale_factors
        self.verbose = verbose
        self.timesteps_per_day = config.timesteps_per_day
        self.optimizers = config.optimizers if hasattr(config, 'optimizers') else ["L-BFGS-B"]
    
    def run(self, truth_data_15ch, clean_truth_15ch, base_state, base_params, metapop_handle):
        """
        Main GSS optimization routine with multi-optimizer support
        
        Returns:
            best_results: dict with keys per optimizer ({"L-BFGS-B": {...}, "CG": {...}})
            struct: theta structure
            all_attempts: list of all optimization attempts across all offsets and optimizers
        """
        import flu_core as flu
        
        struct = build_gss_theta_structure(self.estimation_config, base_state, base_params)
        truth_agg = truth_data_15ch.sum(dim=2, keepdim=True)  # Aggregate over age for Stage 1
        
        # Build regularization terms
        L, A, R = base_params.beta_baseline.shape
        shape_dict = {"beta": (L,)}
        for comp_name, do_est in self.estimation_config["estimate_initial"].items():
            if do_est:
                shape_dict[comp_name] = (L, A, R)
        
        reg_terms = build_regularization_terms(self.config, shape_dict)
        loss_fn_obj = RegionalLossFunction(self.config, aggregation_mode="regional", timesteps_per_day=self.timesteps_per_day)
        
        # Store results per optimizer
        optimizer_results = {opt_name: [] for opt_name in self.optimizers}
        all_attempts = []
        
        # GSS parameters
        a, b = self.config.gss_offset_range if self.config.enable_gss else (0, 0)
        inv_phi2, inv_phi = (3 - np.sqrt(5)) / 2, (np.sqrt(5) - 1) / 2
        
        if not self.config.enable_gss:
            # No offset search, just optimize at offset=0
            offsets_to_probe = [0]
        else:
            c, d = a + inv_phi2 * (b - a), a + inv_phi * (b - a)
            offsets_to_probe = [a, b, c, d]
        
        def evaluate_offset_with_optimizer(offset, optimizer_name):
            """Run one optimizer at one offset"""
            start_time = global_time.time()
            offset = int(round(offset))
            
            # Check if already computed
            for prev_offset, prev_res in optimizer_results[optimizer_name]:
                if prev_offset == offset:
                    return prev_res
            
            print(f"\n{'-'*60}")
            print(f"OPTIMIZER: {optimizer_name} | OFFSET: {offset} days")
            print(f"{'-'*60}")
            
            # Warm-start from nearest offset if available
            if optimizer_results[optimizer_name]:
                closest = min(optimizer_results[optimizer_name], key=lambda x: abs(x[0] - offset))
                x0 = closest[1]['theta_opt'].copy()
                print(f"Warm-starting from offset {closest[0]}")
            else:
                x0 = np.zeros(struct["size"])
                L, A, R = base_params.beta_baseline.shape
                slices = struct["slices"]
                
                # Beta initialization
                if "beta" in slices:
                    s_beta = slices["beta"]
                    # Use true beta values with jitter as initial guess
                    true_betas = base_params.beta_baseline[:, 0, 0].detach().cpu().numpy()
                    jitter = np.random.uniform(0.8, 1.2, size=len(true_betas))
                    beta_init = true_betas * jitter
                    x0[s_beta] = np.log(beta_init * self.scale_factors["beta"])
                    print(f"Initialized beta from true values: {beta_init}")
                
                # E0 initialization
                if "init_E" in slices:
                    s_e = slices["init_E"]
                    # Initialize with small value in seeded location, near-zero elsewhere
                    L, A, R = base_params.beta_baseline.shape
                    e0_init = np.zeros(L * A * R)
                    # Seed location 1, age 2 (index 1*5 + 2 = 7). Subject to change
                    seed_idx = 1 * A + 2  # Loc 1, Age 2
                    e0_init[seed_idx] = 1.0  # Single seeded compartment
                    x0[s_e] = np.log((e0_init + 1e-12) * self.scale_factors["E"])
                    print(f"Initialized E0 with seed at location 1, age 2")
                
                # Initialize other compartments near zero
                for comp_name in ["IP", "ISR", "ISH", "IA"]:
                    key = f"init_{comp_name}"
                    if key in slices:
                        s_comp = slices[key]
                        comp_init = np.full(s_comp.stop - s_comp.start, 1e-8)
                        x0[s_comp] = np.log(comp_init * self.scale_factors.get(comp_name, 1.0))
                        
            # Shift truth data
            if offset >= 0:
                shifted = truth_agg[offset:]
                shifted_clean = clean_truth_15ch[offset:]
                current_T = self.config.T - offset
            else:
                pad_n = torch.zeros((abs(offset), 3, 1, 1))
                pad_c = torch.zeros((abs(offset), 3, 5, 1))
                shifted = torch.cat([pad_n, truth_agg], dim=0)[:self.config.T]
                shifted_clean = torch.cat([pad_c, clean_truth_15ch], dim=0)[:self.config.T]
                current_T = self.config.T
            
            # Build loss function for this offset
            tracker = [0]
            
            L, A, R = base_params.beta_baseline.shape
            
            def loss_fn(x_np):
                """Loss function that returns (loss, grad)"""
                theta = torch.from_numpy(x_np).to(torch.float64)
                theta = torch.clamp(theta, min=-15.0, max=15.0)
                theta = theta.detach().requires_grad_(True)
                init_s, par = apply_gss_theta(theta, self.estimation_config, struct, base_state, base_params, self.scale_factors)
                
                inputs = metapop_handle.get_flu_torch_inputs()
                pred = flu.torch_simulate_hospital_admits(
                    init_s, par, inputs["precomputed"], inputs["schedule_tensors"], 
                    current_T, self.timesteps_per_day
                )
                
                # Compute regularization
                reg_dict = {}
                theta_dict_natural = {}
                
                # Beta in natural scale
                if "beta" in struct["slices"]:
                    s_beta = struct["slices"]["beta"]
                    theta_dict_natural["beta"] = torch.exp(theta[s_beta]) / self.scale_factors.get("beta", 1.0)
                
                # Compartments in natural scale
                for comp_name, do_est in self.estimation_config["estimate_initial"].items():
                    if do_est and f"init_{comp_name}" in struct["slices"]:
                        s_comp = struct["slices"][f"init_{comp_name}"]
                        comp_scale = self.scale_factors.get(comp_name, 1.0)
                        # L, A, R are now defined in enclosing scope
                        theta_dict_natural[comp_name] = (torch.exp(theta[s_comp]) / comp_scale).view(L, A, R)
                
                total_reg = torch.tensor(0.0, dtype=torch.float64)
                for reg_name, reg_term in reg_terms.items():
                    reg_val = reg_term.compute(theta_dict_natural)
                    reg_dict[reg_name] = reg_val
                    total_reg += reg_val
                
                # Compute data fit loss
                loss_components = loss_fn_obj(pred, shifted)
                fit_loss = torch.tensor(loss_components.sse, dtype=torch.float64)
                
                total_loss = fit_loss + total_reg
                total_loss.backward()
                
                # Report progress
                grad_norm = torch.norm(theta.grad).item() if theta.grad is not None else 0.0
                if self.verbose:
                    print(f"Iter {tracker[0]:03d} | Loss: {total_loss.item():.4f} "
                        f"(SSE: {fit_loss.item():.4f}, Reg: {total_reg.item():.6e}), "
                        f"R²: {loss_components.global_r2:.4f}, GradNorm: {grad_norm:.6e}")
                    
                loss_val = total_loss.item()
                if not np.isfinite(loss_val):
                    print(f"ERROR: Non-finite loss at iter {tracker[0]}: {loss_val}")
                    return 1e12, np.zeros_like(x_np)  # Return huge loss to signal optimizer
                
                grad_np = theta.grad.detach().numpy().copy()
                if not np.all(np.isfinite(grad_np)):
                    print(f"WARNING: Non-finite gradients detected at iter {tracker[0]}")
                    grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=1e6, neginf=-1e6)
                
                tracker[0] += 1
                return loss_val, grad_np
            # Run optimizer
            if optimizer_name == "L-BFGS-B":
                opts = {
                    'gtol': 1e-05,      
                    'ftol': 1e-09,     
                    'maxiter': 1000,    
                    'maxfun': 15000   
                }
                if self.verbose:
                    opts['iprint'] = 1
                try:
                    res = minimize(
                        lambda x: loss_fn(x)[0],
                        x0,
                        jac=lambda x: loss_fn(x)[1],
                        method='L-BFGS-B',
                        options=opts
                    )
                except Exception as e:
                    print(f"ERROR: {optimizer_name} failed with: {e}")
                    # Return dummy result with high loss
                    res = type('Result', (), {
                        'x': x0,
                        'fun': 1e12,
                        'success': False,
                        'message': f'Failed: {e}',
                        'nit': 0
                    })()
            
            elif optimizer_name == "CG":
                opts = {'gtol': 1e-04}
                if self.verbose:
                    opts['disp'] = True
                
                try:
                    res = minimize(
                        lambda x: loss_fn(x)[0],
                        x0,
                        jac=lambda x: loss_fn(x)[1],
                        method='CG',
                        options=opts
                    )
                except Exception as e:
                    print(f"ERROR: {optimizer_name} failed with: {e}")
                    # Return dummy result with high loss
                    res = type('Result', (), {
                        'x': x0,
                        'fun': 1e12,
                        'success': False,
                        'message': f'Failed: {e}',
                        'nit': 0
                    })()
                
            
            elif optimizer_name == "Adam":
                # Manual Adam implementation
                theta = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
                adam_optimizer = torch.optim.Adam([theta], lr=0.01)
                
                for i in range(1000):
                    adam_optimizer.zero_grad()
                    loss_val, grad_np = loss_fn(theta.detach().numpy())
                    if not np.isfinite(loss_val):
                        break
                    theta.grad = torch.tensor(grad_np)
                    adam_optimizer.step()
                
                try:
                    res = type('Result', (), {
                        'x': theta.detach().numpy(),
                        'fun': loss_val,
                        'success': True,
                        'message': 'Adam completed',
                        'nit': 1000
                    })()
                except Exception as e:
                    print(f"ERROR: {optimizer_name} failed with: {e}")
                    # Return dummy result with high loss
                    res = type('Result', (), {
                        'x': x0,
                        'fun': 1e12,
                        'success': False,
                        'message': f'Failed: {e}',
                        'nit': 0
                    })()
            
            elif optimizer_name == "least_squares_fd":
                from scipy.optimize import least_squares
                
                def residuals(x_np):
                    loss_val, _ = loss_fn(x_np)
                    return np.array([np.sqrt(max(loss_val, 0.0))])
                
                try:
                    res = least_squares(residuals, x0, jac='2-point', max_nfev=1000)
                except Exception as e:
                    print(f"ERROR: {optimizer_name} failed with: {e}")
                    # Return dummy result with high loss
                    res = type('Result', (), {
                        'x': x0,
                        'fun': 1e12,
                        'success': False,
                        'message': f'Failed: {e}',
                        'nit': 0
                    })()
                res.fun = res.cost * 2  # Convert back to loss
            
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
            print(f"{optimizer_name} Termination: {res.message}")
            
            # Evaluate final solution
            with torch.no_grad():
                theta_final = torch.from_numpy(res.x)
                init_s, par = apply_gss_theta(theta_final, self.estimation_config, struct, base_state, base_params, self.scale_factors)
                inputs = metapop_handle.get_flu_torch_inputs()
                final_p = flu.torch_simulate_hospital_admits(
                    init_s, par, inputs["precomputed"], inputs["schedule_tensors"],
                    current_T, self.timesteps_per_day
                )
                
                final_components = loss_fn_obj(final_p, shifted)
                
                # Recompute regularization for breakdown
                theta_dict_natural = {}
                if "beta" in struct["slices"]:
                    s_beta = struct["slices"]["beta"]
                    theta_dict_natural["beta"] = torch.exp(theta_final[s_beta]) / self.scale_factors.get("beta", 1.0)
                
                for comp_name, do_est in self.estimation_config["estimate_initial"].items():
                    if do_est and f"init_{comp_name}" in struct["slices"]:
                        s_comp = struct["slices"][f"init_{comp_name}"]
                        comp_scale = self.scale_factors.get(comp_name, 1.0)
                        theta_dict_natural[comp_name] = (torch.exp(theta_final[s_comp]) / comp_scale).view(L, A, R)
                
                reg_breakdown = {}
                total_reg_val = 0.0
                for reg_name, reg_term in reg_terms.items():
                    reg_val = reg_term.compute(theta_dict_natural).item()
                    reg_breakdown[reg_name] = reg_val
                    total_reg_val += reg_val
                
                if total_reg_val < 1e-10 and tracker[0] == 0:
                    print(f"WARNING: Total regularization is very small ({total_reg_val:.6e}). "
                          f"Check lambda values and parameter scales.")
                
                save_regional_aggregate_plot(
                    shifted, shifted_clean, final_p, current_T,
                    filename=f"fit_{optimizer_name}_offset_{offset}.png"
                )
            duration = global_time.time() - start_time
            result = {
                'optimizer': optimizer_name,
                'offset': offset,
                'T': current_T,
                'loss': res.fun,
                'theta_opt': res.x,
                'pure_fit_sse': final_components.sse,
                'reg_breakdown': reg_breakdown,
                'total_reg': total_reg_val,
                'reg_sse': final_components.regional_sse,
                'reg_r2': final_components.regional_r2,
                'global_r2': final_components.global_r2,
                'nit': getattr(res, 'nit', 0),
                'duration': duration
            }
            
            optimizer_results[optimizer_name].append((offset, result))
            all_attempts.append(result)
            
            return result
        
        # Run GSS or single offset evaluation
        if not self.config.enable_gss:
            # Just evaluate offset=0 with all optimizers
            for opt_name in self.optimizers:
                evaluate_offset_with_optimizer(0, opt_name)
        else:
            # Full GSS for each optimizer
            for opt_name in self.optimizers:
                print(f"\n{'='*70}")
                print(f"GOLDEN SECTION SEARCH FOR {opt_name}")
                print(f"{'='*70}")
                
                # Initial probes
                for off in [a, b]:
                    evaluate_offset_with_optimizer(off, opt_name)
                
                yc = evaluate_offset_with_optimizer(c, opt_name)['loss']
                yd = evaluate_offset_with_optimizer(d, opt_name)['loss']
                
                # GSS iterations
                a_local, b_local, c_local, d_local = a, b, c, d
                while (b_local - a_local) > self.config.gss_tolerance:
                    print(f"\nGSS Interval for {opt_name}: [{a_local:.2f}, {b_local:.2f}] (width: {(b_local-a_local):.2f})")
                    
                    if yc < yd:
                        print(f"  f(c)={yc:.3f} < f(d)={yd:.3f}. Discarding [{d_local:.2f}, {b_local:.2f}]")
                        b_local, d_local, yd = d_local, c_local, yc
                        c_local = a_local + inv_phi2 * (b_local - a_local)
                        yc = evaluate_offset_with_optimizer(c_local, opt_name)['loss']
                    else:
                        print(f"  f(d)={yd:.3f} <= f(c)={yc:.3f}. Discarding [{a_local:.2f}, {c_local:.2f}]")
                        a_local, c_local, yc = c_local, d_local, yd
                        d_local = a_local + inv_phi * (b_local - a_local)
                        yd = evaluate_offset_with_optimizer(d_local, opt_name)['loss']
        
        # Select best result per optimizer
        best_results = {}
        for opt_name in self.optimizers:
            if optimizer_results[opt_name]:
                best_res = min(optimizer_results[opt_name], key=lambda x: x[1]['loss'])[1]
                best_results[opt_name] = best_res
                
                # Generate convergence plot for this optimizer
                if self.config.enable_gss:
                    save_convergence_plot(
                        [(off, res) for off, res in optimizer_results[opt_name]],
                        filename=f"SSE_Stage1_Convergence_{opt_name}.png"
                    )
        
        return best_results, struct, all_attempts
