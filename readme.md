# CLT Calibration Framework

This framework enables the calibration of the CLT metapopulation model (based on the work of Remy and the [CLT_BaseModel](https://github.com/RemyFP/CLT_BaseModel)) to epidemiological data. It extends the base model with modular optimization routines, regional loss decomposition, and advanced parameter estimation techniques (GSS, Time Stretching) designed specifically for modeling epidemic dynamics in metropolitan areas (e.g., Austin, TX).

## Installation and Usage

### Prerequisites
- Python 3.8+
- [CLT_BaseModel](https://github.com/RemyFP/CLT_BaseModel): The framework uses the `CLT_BaseModel` package. Ensure relevant JSON configuration files (e.g., `ABC_mixing_params.json` for the 3-region example) are present in `CLT_BaseModel/flu_instances/calibration_research_input_files`.

### Quick Start Instructions

Run the following commands to set up the environment and execute the examples:

```bash
# 1. Clone the repository
git clone https://github.com/paul821/clt-calibration-pipeline.git
cd clt-calibration-pipeline

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -e .

# 4. Run Example: Austin Two-Region Calibration (Time Stretch)
python examples/austin_2region_tester.py

# 5. Commit your changes
# git add .
# git commit -m "Your commit message"
# git push
```

## Calibration Modes

The pipeline treats calibration as a modular problem, separating the estimation of transmission potential ($\beta$), severity (IHR), and temporal dynamics (Time Stretch).

### 1. Beta Estimation Mode (`BETA_ONLY`)

**Motivation**  
Estimates the transmission rate ($\beta$) and initial conditions (seed location/size). Since $\beta$ drives the overall exponential growth but is less sensitive to age-specific variations than IHR, this mode typically targets **age-aggregated** hospitalization data to avoid overfitting initial conditions to age-specific noise.

**Inputs**
- **Data**: Regional Hospitalization Time Series (Age-Aggregated). Shape: `[T x L]` (Time x Locations).
- **Base Model Config**: `ABC_mixing_params.json` (3-region) or `AB_mixing_params.json` (2-region).
- **Fixed Parameters**: IHR (National averages), Initial Immunity.

**Outputs**
- **Estimated Parameters**: 
  - $\beta$ (Scalar per region).
  - Initial Compartment Sizes (e.g., $E_0$, $I_{P,0}$).
- **Plots**: 
  - `calibration_regional_aggregate.png`: Line plot of Aggregated Hospitalizations vs Time (Ground Truth vs Model) for each region.

### 2. IHR Estimation Mode (`IHR_MODE`)

**Motivation**  
Resolves age-specific severity using Emergency Department (ED) data. We often lack direct age-specific hospitalization time series but have ED visits by age. Using the ratio $\rho_{A,R} = \frac{\sum H_{A,R}}{\sum ED_{A,R}}$, we derive proxy hospitalization curves. 

This mode solves the identifiability problem between $\beta$ and IHR by chaining two stages:
1.  **Stage 1**: Run `BETA_ONLY` mode on aggregated data to fix the epidemic curve shape.
2.  **Stage 2**: Fix $\beta$ and estimate IHR per age/region to match the age-stratified derived Data.

**Inputs**
- **Data**: Derived Age-Specific Hospitalization Time Series. Shape: `[T x L x A]` (Time x Locations x Ages).
- **Stage 1 Output**: Optimal $\beta$ and seed conditions.

**Outputs**
- **Estimated Parameters**: IHR Matrix (Shape: `[L x A]`).
- **Plots**: 
  - `calibration_15_panel.png`: Grid of scatter/line plots (Time vs Hospitalizations) for every Location-Age pair (e.g., Loc A - Age 0-4).

**Key Features**:
- **Golden Section Search (GSS)**: Automatically discovers the temporal offset (lag) between model and data to align peaks before calculating loss.
- **Regional Loss**: Sums SSE for each region individually to prevent large-population regions from dominating the fit.

### 3. Time Stretch Estimation (Advanced)

**Motivation**  
Standard SEIR models can sometimes produce epidemic waves that are too "fast" or "peaked" compared to observed COVID-19 or Influenza waves. **Time Stretch** introduces a global scaling factor to the transition rates.
- Factor > 1.0: Slows down dynamics (elongates the wave).
- Factor < 1.0: Speeds up dynamics.

If the time stretch factor is incorrect (e.g., fixed at 1.0 when data suggests 1.1), the optimizer may force $\beta$ to extremely low values or distort IHRs to compensate for the mismatched duration, leading to poor predictive power.

**Inputs**
- `config.estimate_time_stretch = True` (in `CalibrationConfig`).

**Outputs**
- **Time Stretch Factor**: Scalar value (e.g., 1.12).

## Optimizers

The framework supports multiple optimization algorithms to ensure robustness:

- **L-BFGS-B**: Quasi-Newton method. Best for smooth, bounded problems (e.g., finding $\beta$ within $[0, 1]$). Fast convergence but can get stuck in local minima.
- **CG (Conjugate Gradient)**: Gradient-based. Good for large-scale unbounded problems. Often more robust to "flat" landscapes than L-BFGS-B.
- **Adam**: Stochastic Gradient Descent (Adaptive Moment Estimation). Best for noisy objectives or when the loss landscape is rugged. Slower but very reliable.
- **Least Squares (FD)**: Standard Levenberg-Marquardt with finite differences. Good baseline for simple parameter sets.

## Running Examples

Confirm the functionality by running the included examples:

1.  **`examples/basic_calibration.py`**: Runs a simple `BETA_ONLY` fit on synthetic data. Verifies the pipeline can recover a known $\beta$.
2.  **`examples/ihr_mode_example.py`**: Runs the full 2-stage `IHR_MODE` with GSS. Verifies that the pipeline can align time-shifted data and recover age-specific rates.
3.  **`examples/austin_2region_tester.py`**: Runs a simulation on a 2-node "Austin-like" network with Time Stretch. Verifies the model can recover a time stretch factor of 1.1.
4.  **`examples/multi_compartment_example.py`**: Demonstrates calibration of multiple initial compartments (e.g., $E_0$, $I_{P,0}$) simultaneously.

Next Step: Apply the pipeline to real-world NSSP ED data.
