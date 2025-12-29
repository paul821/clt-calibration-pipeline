# Libraries to import:
import torch
import numpy as np
from scipy.optimize import minimize, least_squares
import copy
import multiprocessing as mp
import time as global_time

from ..utils.theta_transforms import build_multi_optimizer_theta_structure, apply_multi_optimizer_theta
from ..loss.regional_loss import RegionalLossFunction
from ..loss.regularization import build_regularization_terms

class MultiOptimizerStage1:
    """
    Multi-optimizer suite for Stage 1 (your approach + professor's improvements)
    
    Used for non-IHR modes: BETA_ONLY, or beta + initial compartments
    
    Features:
    - Multiple optimizers run in parallel or sequentially
    - Restart strategy (Wide/Medium/Narrow)
    - Regional loss decomposition
    - Structured regularization
    - Warm-starting between restart phases
    """
    
    def __init__(self, config, estimation_config, scale_factors):
        self.config = config
        self.estimation_config = estimation_config
        self.scale_factors = scale_factors
        self.timesteps_per_day = config.timesteps_per_day
        self.optimizers = config.optimizers
    
    def run(self, truth_data, clean_truth, base_state, base_params, metapop_handle):
        """
        Run multi-optimizer calibration suite
        
        Returns:
            results_df: pandas DataFrame with all attempts
            best_per_optimizer: dict mapping optimizer name → best result
            structure: theta structure
        """
        import flu_core as flu
        import pandas as pd
        
        struct = build_multi_optimizer_theta_structure(self.estimation_config, base_state, base_params)
        
        # Build regularization
        L, A, R = base_params.beta_baseline.shape
        shape_dict = {}
        if "beta" in struct["slices"]:
            shape_dict["beta"] = (L,)
        for comp_name, do_est in self.estimation_config.get("estimate_initial", {}).items():
            if do_est:
                shape_dict[comp_name] = (L, A, R)
        
        reg_terms = build_regularization_terms(self.config, shape_dict)
        loss_fn_obj = RegionalLossFunction(
            self.config,
            aggregation_mode=self.config.loss_aggregation,
            timesteps_per_day=self.timesteps_per_day
        )
        
        # Build loss function
        def build_loss_fn(observed_data):
            def loss_and_grad(x_np):
                theta = torch.from_numpy(x_np).to(torch.float64).detach().requires_grad_(True)
                init_s, par = apply_multi_optimizer_theta(
                    theta, self.estimation_config, struct, base_state, base_params, self.scale_factors
                )
                
                inputs = metapop_handle.get_flu_torch_inputs()
                pred = flu.torch_simulate_hospital_admits(
                    init_s, par, inputs["precomputed"], inputs["schedule_tensors"],
                    self.config.T, self.timesteps_per_day
                )
                
                # Regularization
                theta_dict_natural = {}
                if "beta" in struct["slices"]:
                    s_beta = struct["slices"]["beta"]
                    theta_dict_natural["beta"] = torch.exp(theta[s_beta]) / self.scale_factors.get("beta", 1.0)
                
                for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
                    key = f"init_{comp_name}"
                    if key in struct["slices"]:
                        s_comp = struct["slices"][key]
                        comp_scale = self.scale_factors.get(comp_name, 1.0)
                        theta_dict_natural[comp_name] = (torch.exp(theta[s_comp]) / comp_scale).view(L, A, R)
                
                total_reg = torch.tensor(0.0, dtype=torch.float64)
                for reg_name, reg_term in reg_terms.items():
                    total_reg += reg_term.compute(theta_dict_natural)
                
                # Data fit
                loss_components = loss_fn_obj(pred, observed_data)
                fit_loss = torch.tensor(loss_components.sse, dtype=torch.float64)
                
                total_loss = fit_loss + total_reg
                total_loss.backward()
                
                return total_loss.item(), theta.grad.detach().numpy().copy(), loss_components.global_r2
            
            return loss_and_grad
        
        loss_fn = build_loss_fn(truth_data)
        
        # Generate initial guesses
        def make_initial_guess(randomize=False):
            theta0 = np.zeros(struct["size"], dtype=float)
            
            if "beta" in struct["slices"]:
                s_beta = struct["slices"]["beta"]
                n_beta = s_beta.stop - s_beta.start
                if randomize:
                    beta0 = np.random.uniform(0.003, 0.006, size=n_beta)
                else:
                    beta0 = np.full(n_beta, 0.005)
                theta0[s_beta] = np.log(beta0 * self.scale_factors.get("beta", 1.0))
            
            for comp_name in ["E", "IP", "ISR", "ISH", "IA"]:
                key = f"init_{comp_name}"
                if key not in struct["slices"]:
                    continue
                s_comp = struct["slices"][key]
                base_comp = getattr(base_state, comp_name)
                comp_scale = self.scale_factors.get(comp_name, 1.0)
                comp0 = base_comp.detach().cpu().numpy().reshape(-1).astype(float)
                comp0 = np.clip(comp0, 1e-12, None)
                
                if randomize:
                    jitter = np.random.uniform(0.8, 1.25, size=comp0.shape[0])
                    theta0[s_comp] = np.log((comp0 * jitter) * comp_scale)
                else:
                    theta0[s_comp] = np.log(comp0 * comp_scale)
                
            return theta0
    
        initial_guesses = [
            make_initial_guess(randomize=False),
            make_initial_guess(randomize=True)
        ]
        
        # Run optimizers with restarts
        all_attempts = []
        
        for opt_name in self.optimizers:
            print(f"\n{'='*70}")
            print(f"RUNNING OPTIMIZER: {opt_name}")
            print(f"{'='*70}")
            
            for guess_id, x0 in enumerate(initial_guesses, 1):
                print(f"\nInitial Guess {guess_id}/{len(initial_guesses)}")
                
                # Initial attempt
                result_initial = self._run_single_attempt(
                    opt_name, x0, loss_fn, guess_id, "Initial", 0
                )
                all_attempts.append(result_initial)
                
                best_so_far = result_initial
                
                # Check early stop
                if result_initial['r_squared'] >= self.config.early_stop_r2:
                    print(f"Early stop: R² = {result_initial['r_squared']:.4f} >= {self.config.early_stop_r2}")
                    continue
                
                # Restart phases
                restart_phases = [
                    ("Wide Search", self.config.num_wide_restarts, self.config.restart_widths["Wide Search"]),
                    ("Medium Search", self.config.num_medium_restarts, self.config.restart_widths["Medium Search"]),
                    ("Narrow Search", self.config.num_narrow_restarts, self.config.restart_widths["Narrow Search"])
                ]
                
                for phase_name, num_restarts, width in restart_phases:
                    for restart_idx in range(num_restarts):
                        print(f"\n{phase_name} - Restart {restart_idx+1}/{num_restarts}")
                        
                        # Generate restart point around best
                        restart_x0 = self._generate_restart_point(
                            best_so_far['theta_opt'], struct, width
                        )
                        
                        result_restart = self._run_single_attempt(
                            opt_name, restart_x0, loss_fn, guess_id, phase_name, restart_idx+1
                        )
                        all_attempts.append(result_restart)
                        
                        if result_restart['loss'] < best_so_far['loss']:
                            best_so_far = result_restart
                        
                        if result_restart['r_squared'] >= self.config.early_stop_r2:
                            print(f"Early stop in restart: R² >= {self.config.early_stop_r2}")
                            break
                    
                    if result_restart['r_squared'] >= self.config.early_stop_r2:
                        break
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_attempts)
        
        # Best per optimizer
        best_per_optimizer = {}
        for opt_name in self.optimizers:
            opt_results = results_df[results_df['optimizer'] == opt_name]
            if not opt_results.empty:
                best_idx = opt_results['loss'].idxmin()
                best_per_optimizer[opt_name] = results_df.loc[best_idx].to_dict()
        
        return results_df, best_per_optimizer, struct

    def _run_single_attempt(self, optimizer_name, x0, loss_fn, guess_id, phase, restart_num):
        """Run one optimization attempt"""
        start_time = global_time.time()
        
        if optimizer_name == "L-BFGS-B":
            res = minimize(
                lambda x: loss_fn(x)[0],
                x0,
                jac=lambda x: loss_fn(x)[1],
                method='L-BFGS-B',
                options={'gtol': 1e-04, 'ftol': 1e-07}
            )
        
        elif optimizer_name == "CG":
            res = minimize(
                lambda x: loss_fn(x)[0],
                x0,
                jac=lambda x: loss_fn(x)[1],
                method='CG',
                options={'gtol': 1e-04}
            )
        
        elif optimizer_name == "Adam":
            theta = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            adam_optimizer = torch.optim.Adam([theta], lr=0.01)
            
            final_loss = 1e12
            for i in range(1000):
                adam_optimizer.zero_grad()
                loss_val, grad_np, r2 = loss_fn(theta.detach().numpy())
                if not np.isfinite(loss_val):
                    break
                theta.grad = torch.tensor(grad_np)
                adam_optimizer.step()
                final_loss = loss_val
            
            res = type('Result', (), {
                'x': theta.detach().numpy(),
                'fun': final_loss,
                'success': True,
                'nit': 1000
            })()
        
        elif optimizer_name == "least_squares_fd":
            def residuals(x_np):
                loss_val, _, _ = loss_fn(x_np)
                return np.array([np.sqrt(max(loss_val, 0.0))])
            
            res = least_squares(residuals, x0, jac='2-point', max_nfev=1000)
            res.fun = res.cost * 2
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        duration = global_time.time() - start_time
        
        # Final evaluation
        _, _, final_r2 = loss_fn(res.x)
        
        return {
            'optimizer': optimizer_name,
            'initial_guess_id': guess_id,
            'phase': phase,
            'restart_num': restart_num,
            'loss': res.fun,
            'r_squared': final_r2,
            'theta_opt': res.x,
            'duration': duration,
            'nit': getattr(res, 'nit', 0)
        }

    def _generate_restart_point(self, base_theta, struct, width):
        """Generate restart point with noise"""
        theta_new = base_theta.copy()
        
        # Add noise to beta
        if "beta" in struct["slices"]:
            s_beta = struct["slices"]["beta"]
            beta_natural = np.exp(base_theta[s_beta]) / self.scale_factors.get("beta", 1.0)
            low = beta_natural * (1.0 - width)
            high = beta_natural * (1.0 + width)
            low = np.maximum(low, 1e-8)
            perturbed_beta = np.random.uniform(low, high)
            theta_new[s_beta] = np.log(perturbed_beta * self.scale_factors.get("beta", 1.0))
        
        return theta_new