# Libraries to import:
import torch
import numpy as np
from scipy.optimize import minimize, least_squares
import copy
import multiprocessing as mp
import time as global_time
import pandas as pd

from ..utils.theta_transforms import apply_ihr_theta
from ..loss.regional_loss import RegionalLossFunction

class MultiOptimizerStage2:
    """
    Multi-optimizer suite for Stage 2 IHR estimation (your approach)
    
    Used when Stage 1 beta/initial compartments are already fixed.
    
    Features:
    - Multiple optimizers run in parallel or sequentially
    - Restart strategy (Wide/Medium/Narrow) with tighter bounds for IHR
    - Location-age loss aggregation (15-channel)
    - Warm-starting between restart phases
    """
    
    def __init__(self, config, scale_factors):
        self.config = config
        self.scale_factors = scale_factors
        self.timesteps_per_day = config.timesteps_per_day
        self.optimizers = config.optimizers
        self.ihr_param = config.estimation_config.get("ihr_param", "L")
    
    def run(self, truth_15ch, state_t0, params_fixed_beta, metapop, current_T):
        """
        Run multi-optimizer IHR calibration suite
        
        Args:
            truth_15ch: truth data (T, L, A, R)
            state_t0: initial state (with fixed beta from Stage 1)
            params_fixed_beta: params with beta fixed from Stage 1
            metapop: metapopulation model
            current_T: simulation time horizon
        
        Returns:
            results_df: pandas DataFrame with all attempts
            best_per_optimizer: dict mapping optimizer name → best result
        """
        import flu_core as flu
        import clt_toolkit as clt
        
        # Initialize IHR guess
        x0_natural = params_fixed_beta.IP_to_ISH_prop.detach().numpy()
        L, A, R = params_fixed_beta.beta_baseline.shape
        
        if self.ihr_param == "L":
            # Average across age and risk to get per-location IHR
            x0_natural = x0_natural.reshape(L, A, R).mean(axis=(1, 2))
            n_ihr = L
        elif self.ihr_param == "LAR":
            x0_natural = x0_natural.flatten()
            n_ihr = L * A * R
        else:
            raise ValueError(f"ihr_param must be 'L' or 'LAR', got {self.ihr_param}")
        
        # Convert to log-space
        x0_log = np.log(x0_natural * self.scale_factors.get("ihr", 1.0))
        
        # Build loss function
        loss_fn_obj = RegionalLossFunction(
            self.config,
            aggregation_mode="location_age",  # 15-channel for IHR
            timesteps_per_day=self.timesteps_per_day
        )
        
        def build_loss_fn():
            def loss_and_grad(x_np):
                theta = torch.from_numpy(x_np).to(torch.float64).detach().requires_grad_(True)
                par = apply_ihr_theta(theta, params_fixed_beta, self.scale_factors)
                
                # Update metapop IHR
                metapop._full_metapop_params_tensors.IP_to_ISH_prop = par.IP_to_ISH_prop.detach()
                for i, sub in enumerate(metapop.subpop_models.values()):
                    sub.params = clt.updated_dataclass(
                        sub.params,
                        {"IP_to_ISH_prop": par.IP_to_ISH_prop.detach()[i]}
                    )
                
                inputs = metapop.get_flu_torch_inputs()
                pred = flu.torch_simulate_hospital_admits(
                    state_t0, par, inputs["precomputed"], inputs["schedule_tensors"],
                    current_T, self.timesteps_per_day
                )
                
                # Location-age loss
                loss_components = loss_fn_obj(pred, truth_15ch)
                fit_loss = torch.tensor(loss_components.sse, dtype=torch.float64)
                
                fit_loss.backward()
                
                return fit_loss.item(), theta.grad.detach().numpy().copy(), loss_components.global_r2
            
            return loss_and_grad
        
        loss_fn = build_loss_fn()
        
        # Generate initial guesses
        def make_initial_guess(randomize=False):
            if randomize:
                jitter = np.random.uniform(0.8, 1.25, size=n_ihr)
                theta0 = np.log((x0_natural * jitter) * self.scale_factors.get("ihr", 1.0))
            else:
                theta0 = x0_log.copy()
            return theta0
        
        initial_guesses = [
            make_initial_guess(randomize=False),
            make_initial_guess(randomize=True)
        ]
        
        # IHR restart widths (narrower than beta)
        restart_phases = [
            ("Wide Search", self.config.num_wide_restarts, self.config.ihr_restart_widths["Wide Search"]),
            ("Medium Search", self.config.num_medium_restarts, self.config.ihr_restart_widths["Medium Search"]),
            ("Narrow Search", self.config.num_narrow_restarts, self.config.ihr_restart_widths["Narrow Search"])
        ]
        
        # IHR bounds in log-space
        ihr_bounds = [(np.log(0.001 * self.scale_factors.get("ihr", 1.0)), 
                       np.log(0.50 * self.scale_factors.get("ihr", 1.0)))] * n_ihr
        
        # Run optimizers with restarts
        all_attempts = []
        
        for opt_name in self.optimizers:
            print(f"\n{'='*70}")
            print(f"STAGE 2 IHR - OPTIMIZER: {opt_name}")
            print(f"{'='*70}")
            
            for guess_id, x0 in enumerate(initial_guesses, 1):
                print(f"\nInitial Guess {guess_id}/{len(initial_guesses)}")
                
                # Initial attempt
                result_initial = self._run_single_attempt(
                    opt_name, x0, loss_fn, guess_id, "Initial", 0, ihr_bounds
                )
                all_attempts.append(result_initial)
                
                best_so_far = result_initial
                
                # Check early stop
                if result_initial['r_squared'] >= self.config.early_stop_r2:
                    print(f"Early stop: R² = {result_initial['r_squared']:.4f} >= {self.config.early_stop_r2}")
                    continue
                
                # Restart phases
                for phase_name, num_restarts, width in restart_phases:
                    for restart_idx in range(num_restarts):
                        print(f"\n{phase_name} - Restart {restart_idx+1}/{num_restarts}")
                        
                        # Generate restart point
                        restart_x0 = self._generate_restart_point(
                            best_so_far['theta_opt'], width
                        )
                        
                        result_restart = self._run_single_attempt(
                            opt_name, restart_x0, loss_fn, guess_id, phase_name, restart_idx+1, ihr_bounds
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
        
        # Add IHR values in natural scale
        results_df['ihr_values'] = results_df['theta_opt'].apply(
            lambda theta: (np.exp(theta) / self.scale_factors.get("ihr", 1.0)).tolist()
        )
        
        # Best per optimizer
        best_per_optimizer = {}
        for opt_name in self.optimizers:
            opt_results = results_df[results_df['optimizer'] == opt_name]
            if not opt_results.empty:
                best_idx = opt_results['loss'].idxmin()
                best_per_optimizer[opt_name] = results_df.loc[best_idx].to_dict()
        
        return results_df, best_per_optimizer
    
    def _run_single_attempt(self, optimizer_name, x0, loss_fn, guess_id, phase, restart_num, ihr_bounds):
        """Run one optimization attempt"""
        start_time = global_time.time()
        
        if optimizer_name == "L-BFGS-B":
            res = minimize(
                lambda x: loss_fn(x)[0],
                x0,
                jac=lambda x: loss_fn(x)[1],
                method='L-BFGS-B',
                bounds=ihr_bounds,
                options={'gtol': 1e-04, 'ftol': 1e-07}
            )
        
        elif optimizer_name == "CG":
            # CG doesn't support bounds, so we just use unconstrained
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
                
                # Manual bound enforcement for Adam
                with torch.no_grad():
                    for j, (lb, ub) in enumerate(ihr_bounds):
                        theta[j] = torch.clamp(theta[j], min=lb, max=ub)
                
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
            
            # least_squares supports bounds
            lb = np.array([b[0] for b in ihr_bounds])
            ub = np.array([b[1] for b in ihr_bounds])
            res = least_squares(residuals, x0, jac='2-point', bounds=(lb, ub), max_nfev=1000)
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
    
    def _generate_restart_point(self, base_theta, width):
        """Generate restart point with noise around current best"""
        ihr_natural = np.exp(base_theta) / self.scale_factors.get("ihr", 1.0)
        low = ihr_natural * (1.0 - width)
        high = ihr_natural * (1.0 + width)
        low = np.maximum(low, 1e-8)
        
        perturbed_ihr = np.random.uniform(low, high)
        theta_new = np.log(perturbed_ihr * self.scale_factors.get("ihr", 1.0))
        
        return theta_new