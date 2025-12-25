from clt_calib.environment import setup_environment
from clt_calib.config import MODE
from clt_calib.data import build_simulation_inputs, generate_truth
from clt_calib.stage1_gss import run_gss_stage
from clt_calib.stage2_ihr import run_ihr_stage

def main():
    setup_environment()

    inputs = build_simulation_inputs()
    truth = generate_truth(inputs)

    if MODE in ("BETA_ONLY", "SEQUENTIAL"):
        stage1_results = run_gss_stage(inputs, truth)

    if MODE in ("IHR_ONLY", "SEQUENTIAL"):
        run_ihr_stage(inputs, truth, stage1_results)

if __name__ == "__main__":
    main()
