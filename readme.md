# CLT Calibration Framework

This framework enables the calibration of the CLT metapopulation model designed for modeling epidemic dynamics in metropolitan areas like Austin, TX (based on the [CLT_BaseModel](https://github.com/RemyFP/CLT_BaseModel)).

## Installation and Usage

### Prerequisites
- Python 3.8+
- [CLT_BaseModel](https://github.com/RemyFP/CLT_BaseModel): The framework uses the `CLT_BaseModel` package.

### Quick Start Instructions

Run the following commands to set up the environment and execute the examples:

```bash
git clone https://github.com/paul821/clt-calibration-pipeline.git
cd clt-calibration-pipeline
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

pip install -e .

# running examples
python examples/austin_2region_tester.py

# for committing
git add .
git commit -m "Insert comment"
git push
```

## Calibration Modes

The pipeline treats calibration as a modular problem, separating the estimation of infection rates ($\beta$), a proxy for severity (IHR), and temporal dynamics (Time Stretch Factor).

### 1. Beta Estimation Mode (`BETA_ONLY`)

**Motivation**  
Estimates the transmission rate ($\beta$) and initial conditions (seed location/size). This mode targets hospitalization data aggregated over age.

**Inputs**
- Data: Regional Hospitalization Time Series (Age-Aggregated). Shape: `[T x L]` (Time x Locations).
- Base Model Config: `ABC_mixing_params.json` (3-region) or `AB_mixing_params.json` (2-region).
- Fixed Parameters: IHR, Initial Immunity.

**Outputs**
- Estimated Parameters: 
  - $\beta$ (Scalar per region).
  - Initial Compartment Sizes ($E_0$, $I_{P,0}$, etc.).
- **Plots**: 
  - `calibration_regional_aggregate.png`: Line plot of Aggregated Hospitalizations vs Time (Ground Truth vs Model) for each region.

### 2. IHR Estimation Mode (`IHR_MODE`)

**Motivation**  
Resolves age-specific severity using Emergency Department (ED) data. We often lack direct age-specific hospitalization time series but have ED visits by age. Using the ratio $\rho_{A,R} = \frac{\sum H_{A,R}}{\sum ED_{A,R}}$, we derive proxy hospitalization data. 

This mode chains two stages:
1.  Run `BETA_ONLY` mode on aggregated data.
2.  Fix $\beta$ and estimate IHR per age/region to match the derived data.

**Inputs**
- Data: Derived Age-Specific Hospitalization Time Series. Shape: `[T x L x A]` (Time x Locations x Ages).
- Stage 1 Output: Optimal $\beta$ and seed conditions.

**Outputs**
- Estimated Parameters: IHR Matrix (Shape: `[L x A]`).
- Plots: 
  - `calibration_15_panel.png`: Grid of scatter/line plots (Time vs Hospitalizations) for every Location-Age pair.

**Key Features**:
- Golden Section Search (GSS): Automatically discovers the time lag between model and data to align peaks before calculating loss.
- Regional Loss: Sums SSE for each region individually to prevent large population regions from dominating the fit.

### 3. Time Stretch Estimation

- Factor > 1.0: Slows down dynamics.
- Factor < 1.0: Speeds up dynamics.

If the time stretch factor is incorrect, the optimizer may force $\beta$ to extremely low values or distort IHRs to compensate for the mismatched duration, leading to poor predictive power.

**Inputs**
- `config.estimate_time_stretch = True` (in `CalibrationConfig`).

**Outputs**
- Time Stretch Factor: Scalar value (e.g., 1.12).

## Optimizers

The framework supports multiple optimization algorithms to ensure robustness:

- L-BFGS-B: Quasi-Newton method. Fast convergence but can get stuck in local minima.
- CG (Conjugate Gradient): Gradient-based. Good for large-scale unbounded problems. Often more robust to flatter landscapes.
- Adam: Stochastic Gradient Descent. Best for noisy objectives or when the loss landscape is rugged.
- Least Squares (FD): With finite differences. Good baseline.

## Running Examples

1.  `examples/basic_calibration.py`: Runs a simple `BETA_ONLY` fit on synthetic data. Verifies the pipeline can recover a known $\beta$.
2.  `examples/ihr_mode_example.py`: Runs the full 2-stage `IHR_MODE`. Verifies that the pipeline can align time-shifted data and recover age-specific rates.
3.  `examples/austin_2region_tester.py`: Runs a simulation on a 2-node "Austin-like" network with Time Stretch. Verifies the model can recover a time stretch factor of 1.1.
4.  `examples/multi_compartment_example.py`: Demonstrates calibration of multiple initial compartments simultaneously.

Next Step: Apply the pipeline to real data.
