# Libraries to import:
import torch
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import copy
import warnings  # Line 30!

# Import flu_core after environment setup
import flu_core as flu

from ..utils.theta_transforms import build_gss_theta_structure, apply_gss_theta
from ..utils.metrics import format_iter_report
from ..visualization.plotting import save_convergence_plot, save_regional_aggregate_plot

class GSSOptimizer:
    """Golden Section Search optimizer for Stage 1 (Beta & E0)"""
    
    def __init__(self, config, estimation_config, reg_config, verbose=False):
        self.config = config
        self.estimation_config = estimation_config
        self.reg_config = reg_config
        self.verbose = verbose
        self.timesteps_per_day = config.timesteps_per_day
    
    def run(self, truth_data_15ch, clean_truth_15ch, base_state, base_params, metapop_handle):
        """
        Lines 282-381: Complete GSS Stage
        """
        # Lines 283-285
        struct = build_gss_theta_structure(self.estimation_config, base_state, base_params)
        grid_results = []
        truth_agg = truth_data_15ch.sum(dim=2, keepdim=True)
        
        # Lines 287-314: gss_loss_fn
        def gss_loss_fn(x_np, shifted_truth_agg, current_T, iteration_tracker):
            theta = torch.from_numpy(x_np).to(torch.float64).detach().requires_grad_(True)
            init_s, par = apply_gss_theta(theta, self.estimation_config, struct, base_state, base_params)
            inputs = metapop_handle.get_flu_torch_inputs()
            pred = flu.torch_simulate_hospital_admits(init_s, par, inputs["precomputed"], inputs["schedule_tensors"], current_T, self.timesteps_per_day)
            regional_sse = [torch.sum((pred[:, i].sum(dim=(1, 2)) - shifted_truth_agg[:, i].sum(dim=(1, 2))) ** 2) for i in range(3)]
            fit_obj = torch.stack(regional_sse).sum()
            e0_penalty = torch.tensor(0.0, dtype=torch.float64)
            if "init_E" in struct["slices"]:
                e0_vals, target_ages = torch.exp(theta[struct["slices"]["init_E"]]).view(3, 5), self.reg_config["target_e0_values"]
                for r in range(3):
                    for a in range(5):
                        w = self.reg_config["lambda_e0_target"] if a == self.reg_config["target_age_idx"] else self.reg_config["lambda_e0_zero"]
                        e0_penalty += w * (e0_vals[r, a] - (target_ages[r] if a == self.reg_config["target_age_idx"] else 0.0))**2
            total_loss = fit_obj + e0_penalty
            total_loss.backward()
            format_iter_report(pred, shifted_truth_agg, [shifted_truth_agg[:, i].sum(dim=(1, 2)) for i in range(3)], iteration_tracker[0], np.linalg.norm(theta.grad.detach().numpy()), fit_obj.item(), verbose=self.verbose)
            iteration_tracker[0] += 1
            return total_loss.item(), theta.grad.detach().numpy().copy()
        
        # Lines 316-356: evaluate_offset
        def evaluate_offset(offset):
            offset = int(round(offset))
            for off, res in grid_results:
                if off == offset: return res['loss']
            print(f"\n" + "-"*40 + f"\n PROBING OFFSET:   {offset} days \n" + "-"*40)
            if grid_results:
                closest_off = min(grid_results, key=lambda x: abs(x[0] - offset))[0]
                print(f"Warm-starting offset {offset} from closest neighbor offset {closest_off}")
                x0 = next(res['theta_opt'] for off, res in grid_results if off == closest_off).copy()
            else: x0 = np.zeros(struct["size"])
            
            if offset >= 0: 
                shifted = truth_agg[offset:]; shifted_clean = clean_truth_15ch[offset:]
                current_T = self.config.T - offset
            else: 
                pad_n = torch.zeros((abs(offset), 3, 1, 1)); pad_c = torch.zeros((abs(offset), 3, 5, 1))
                shifted = torch.cat([pad_n, truth_agg], dim=0)[:self.config.T]
                shifted_clean = torch.cat([pad_c, clean_truth_15ch], dim=0)[:self.config.T]
                current_T = self.config.T
                
            tracker = [0]
            # Lines 334-336: Surgical Edit - Bypassing deprecation warning
            lbfgs_opts = {'gtol': 1e-04, 'ftol': 1e-07}
            if self.verbose: lbfgs_opts['iprint'] = 1

            res = minimize(lambda x: gss_loss_fn(x, shifted, current_T, tracker)[0], x0, jac=lambda x: gss_loss_fn(x, shifted, current_T, tracker)[1], method='L-BFGS-B', options=lbfgs_opts)
            print(f"BETA Optimization Termination Message: {res.message}")
            
            # Lines 340-365: Post-optimization metric calculation
            with torch.no_grad():
                theta_final = torch.from_numpy(res.x)
                init_s, par = apply_gss_theta(theta_final, self.estimation_config, struct, base_state, base_params)
                inputs = metapop_handle.get_flu_torch_inputs()
                final_p = flu.torch_simulate_hospital_admits(init_s, par, inputs["precomputed"], inputs["schedule_tensors"], current_T, self.timesteps_per_day)
                
                # Recalculate component losses for reporting
                reg_sse = [torch.sum((final_p[:, i].sum(dim=(1, 2)) - shifted[:, i].sum(dim=(1, 2))) ** 2).item() for i in range(3)]
                pure_fit_sse = sum(reg_sse)
                
                e0_penalty = 0.0
                if "init_E" in struct["slices"]:
                    e0_vals = torch.exp(theta_final[struct["slices"]["init_E"]]).view(3, 5)
                    target_ages = self.reg_config["target_e0_values"]
                    for r in range(3):
                        for a in range(5):
                            w = self.reg_config["lambda_e0_target"] if a == self.reg_config["target_age_idx"] else self.reg_config["lambda_e0_zero"]
                            e0_penalty += w * (e0_vals[r, a].item() - (target_ages[r] if a == self.reg_config["target_age_idx"] else 0.0))**2

                reg_r2 = [(1.0 - (reg_sse[i]/torch.sum((shifted[:, i].sum(dim=(1, 2)) - torch.mean(shifted[:, i].sum(dim=(1, 2))))**2).item())) for i in range(3)]
                g_true, g_pred = shifted.sum(dim=(1,2,3)), final_p.sum(dim=(1,2,3))
                g_sse = torch.sum((g_pred - g_true)**2).item()
                g_sstot = torch.sum((g_true - torch.mean(g_true))**2).item()
                global_r2 = 1.0 - (g_sse / g_sstot) if g_sstot > 0 else 0.0
                
                save_regional_aggregate_plot(shifted, shifted_clean, final_p, current_T, filename=f"fit_offset_{offset}.png")
                
            grid_results.append((offset, {
                'loss': res.fun, 
                'theta_opt': res.x, 
                'offset': offset, 
                'T': current_T, 
                'reg_sse': reg_sse, 
                'pure_fit_sse': pure_fit_sse, 
                'reg_penalty': e0_penalty, 
                'reg_r2': reg_r2, 
                'global_r2': global_r2
            }))
            return res.fun
    
        # Lines 358-374: GSS algorithm
        a, b, inv_phi2, inv_phi = -30, 15, (3 - np.sqrt(5)) / 2, (np.sqrt(5) - 1) / 2
        c, d = a + inv_phi2 * (b - a), a + inv_phi * (b - a)
        evaluate_offset(a); evaluate_offset(b); yc, yd = evaluate_offset(c), evaluate_offset(d)
        
        while (b - a) > self.config.gss_tolerance:
            print(f"\n--- GSS Interval: [{a:.2f}, {b:.2f}] (width: {(b-a):.2f}) ---")
            if yc < yd:
                print(f"   Result: f(c)={yc:.3f} < f(d)={yd:.3f}. Discarding upper interval [{d:.2f}, {b:.2f}].")
                b, d, yd = d, c, yc
                c = a + inv_phi2 * (b - a)
                yc = evaluate_offset(c)
            else:
                print(f"   Result: f(d)={yd:.3f} <= f(c)={yc:.3f}. Discarding lower interval [{a:.2f}, {c:.2f}].")
                a, c, yc = c, d, yd
                d = a + inv_phi * (b - a)
                yd = evaluate_offset(d)
        
        # Lines 376-381: Finalization
        save_convergence_plot(grid_results)
        best_res = min(grid_results, key=lambda x: x[1]['loss'])[1]
        opt_st, opt_pa = apply_gss_theta(torch.from_numpy(best_res['theta_opt']), self.estimation_config, struct, base_state, base_params)
        
        stage1_report_args = (base_params.beta_baseline[:,0,0].detach().numpy(), opt_pa.beta_baseline[:,0,0].detach().numpy(), base_state.E.detach(), opt_st.E.detach(), best_res)
        
        # Line 380-381
        from ..utils.metrics import print_beta_e0_table
        print_beta_e0_table(*stage1_report_args)
        print(f"\n>>> GSS Complete. Optimal Offset: {best_res['offset']} days. Best SSE: {best_res['loss']:.3f} <<<")
        return best_res, struct, stage1_report_args
