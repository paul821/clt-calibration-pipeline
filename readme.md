# Flu Model Calibration

This repository contains code for calibrating MetroFluSim models using the CLT Toolkit and PyTorch-based optimization.

The code is intended to be run locally in Python 3.11 using VS Code or a standard terminal.

---

## Requirements

- Python 3.11
- pip
- git

---

## Setup (local machine)

Clone this repository:

```bash
git clone https://github.com/paul821/clt-calibration-pipeline/
cd clt-calibration-pipeline

```
---

### MAJOR MODIFICATIONS:
1. TEMPORAL RESCALING:
   - Transition rates (E->I, I->H, etc.) scaled by 'time_stretch' (5.0).
   - Purpose: Elongates epidemic peaks 

2. LOG-SPACE PARAMETERIZATION:
   - Parameters (beta, E0) optimized as exponents: theta = log(param).
   - Purpose: Guarantees parameter positivity and improves L-BFGS-B 
     convergence across disparate magnitudes.

3. RESTRUCTURED REGIONAL LOSS:
   - Shifted from global tensor MSE to a sum of raw regional 
     squared-error (SSE) terms.
   - Purpose: Provides a high-fidelity objective function that 
     captures regional dynamics while remaining unweighted to 
     reflect raw hospitalizations volumes.

4. E0 STRUCTURAL REGULARIZATION:
   - L2 penalty (lambda=10.0) applied to initial infected states.
   - Target: Constrains E0 to specific age groups (Age Index 2) 
     per REG_CONFIG targets.
   - Purpose: Resolves 'Shadow Solutions' where the model fits the 
     curve but misassigns starting population demographics.

5. STOCHASTIC TRUTH ANCHORING:
   - Truth data injected with IID Poisson noise: N ~ Poisson(Lambda).
   - This is now done at the Location-Age resolution and aggregated for Stage 1.
   - Purpose: Provides a realistic, 'rough' landscape that prevents 
     parameter smearing and tests recovery against stochasticity.

6. OPTIMIZATION HYPERPARAMETERS:
   - Algorithm: L-BFGS-B (Stage 1 & 2).
   - Tolerances: ftol=1e-7, gtol=1e-4. Randomly, one has maxiter and the other does not. No reason.
   - Purpose: Early termination upon reaching the noise floor, 
     preventing over-fitting to stochastic fluctuations.

7. GOLDEN SECTION SEARCH (GSS) OFFSET DISCOVERY:
   - Mechanism: Iterative interval reduction search for temporal 
     shift (Delta) within a [-30, 15] day window. This can we changed to 
     (say) [-1,1] for quicker run.
   - Logic: Each 'probe' executes a full L-BFGS-B optimization 
     to find the minimum SSE SUM (Objective) at that offset.
   - Purpose: Decouples the epidemic 'Start Date' from the 
     Transmission Rate (Beta), resolving temporal phase-shift errors.
     In reality we do not know when infections start. This aims to mimic
     that reality. 

8. MULTI-STAGE PARAMETER DECOUPLING:
   - Structure: Stage 1 (Transmission/Seeding) -> Stage 2 (IHR).
   - Constraint: Stage 1 results (Beta, E0, Offset) are frozen when we 
     get to stage 2
   - Purpose: The idea is to get regional hospitalizations correct at first
     and then try to get age-specific results correct while avoiding the
     identifiability issues that come with fitting beta and IHR at the same time. 

9. 15-CHANNEL CLINICAL CALIBRATION (IHR):
   - Mechanism: Optimization across 15 independent IHR values 
     (3 regions x 5 age groups). This is stage 2. 
   - Purpose: Recovers the probability of hospitalization per 
     infected individual across disparate demographic subpopulations.

10. NEIGHBORHOOD WARM-STARTING:
    - Mechanism: GSS probes are initialized using the optimal theta 
      vector from the nearest previously calculated offset.
    - Purpose: Drastically reduces convergence time and ensures 
      stability across the GSS search landscape.
