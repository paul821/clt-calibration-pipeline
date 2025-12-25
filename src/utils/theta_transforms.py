# Libraries to import:
import torch
import copy

# Migrate entire functions:
def build_gss_theta_structure(config, base_state, base_params):
    L, A, R = base_params.beta_baseline.shape
    slices = {}; idx = 0
    slices["beta"] = slice(idx, idx + L); idx += L
    for comp_name, do_est in config["estimate_initial"].items():
        if do_est:
            n_comp = getattr(base_state, comp_name).numel()
            slices[f"init_{comp_name}"] = slice(idx, idx + n_comp); idx += n_comp
    return {"slices": slices, "size": idx}
  
def apply_gss_theta(theta, config, structure, base_state, base_params):
    L, A, R = base_params.beta_baseline.shape
    slices = structure["slices"]
    beta_tensor = torch.exp(theta[slices["beta"]]).view(L, 1, 1).expand(L, A, R)
    params = copy.deepcopy(base_params); params.beta_baseline = beta_tensor
    init_state = copy.deepcopy(base_state)
    if "init_E" in slices:
        init_state.E = torch.exp(theta[slices["init_E"]]).view_as(init_state.E).double()
    return init_state, params
  
def apply_ihr_theta(theta, base_params):
    params = copy.deepcopy(base_params)
    params.IP_to_ISH_prop = torch.as_tensor(theta, dtype=torch.float64).view(3, 5, 1).contiguous()
    return params
