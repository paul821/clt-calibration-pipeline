
def apply_ihr_theta(theta, base_params):
    params = copy.deepcopy(base_params)
    params.IP_to_ISH_prop = torch.as_tensor(theta, dtype=torch.float64).view(3, 5, 1).contiguous()
    return params



def run_ihr_stage(truth_15ch, state_t0, params_in, metapop, current_T):
    x0, iter_tracker = params_in.IP_to_ISH_prop.detach().numpy().flatten(), [0]
    subpop_truths_15ch = [truth_15ch[:, i].sum(dim=(1, 2)) for i in range(3)]
    def ihr_loss_fn(x_np):
        theta = torch.from_numpy(x_np).to(torch.float64).detach().requires_grad_(True)
        par = apply_ihr_theta(theta, params_in)
        metapop._full_metapop_params_tensors.IP_to_ISH_prop = par.IP_to_ISH_prop.detach()
        for i, sub in enumerate(metapop.subpop_models.values()):
            sub.params = clt.updated_dataclass(sub.params, {"IP_to_ISH_prop": par.IP_to_ISH_prop.detach()[i]})
        inputs = metapop.get_flu_torch_inputs()
        pred = flu.torch_simulate_hospital_admits(state_t0, par, inputs["precomputed"], inputs["schedule_tensors"], current_T, timesteps_per_day)
        fit_obj = torch.sum((pred - truth_15ch)**2); fit_obj.backward()
        format_iter_report(pred, truth_15ch, subpop_truths_15ch, iter_tracker[0], np.linalg.norm(theta.grad.detach().numpy()), fit_obj.item())
        iter_tracker[0] += 1
        return fit_obj.item(), theta.grad.detach().numpy().copy()
    
    # Surgical Edit: Bypassing deprecation warning by conditionally adding iprint
    lbfgs_opts = {'gtol': 1e-04, 'ftol': 1e-07}
    if VERBOSE_LBFGS: lbfgs_opts['iprint'] = 1

    res = minimize(lambda x: ihr_loss_fn(x)[0], x0, jac=lambda x: ihr_loss_fn(x)[1], method='L-BFGS-B', bounds=[(0.001, 0.50)]*15, options=lbfgs_opts)
    print(f"IHR Optimization Termination Message: {res.message}")
    return res.x
