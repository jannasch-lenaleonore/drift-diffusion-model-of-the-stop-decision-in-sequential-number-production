# Model of the MS "The decision to stop ‘counting’ shapes numerical production in primates"


MATLAB code for a drift-diffusion model of the *stop decision* in sequential
number production, fitted to and compared against behavioural data from two
macaques performing a counting task.

## The model

Each numerosity step is treated as one second of a bounded accumulation
process (a Ratcliff-style diffusion). The animal "counts" by accumulating
evidence across discrete steps and emits a response when the decision
variable (DV) first crosses a fixed upper bound `θ = 1`:

- **Drift** is drawn fresh per trial from `N(drift_mean, drift_sigma)`.
  `drift_sigma` is the Ratcliff across-trial drift variability `η`. When
  `fix_drift = true`, `drift_sigma = 0`, drift is constant (MS).
- **Starting point** is a folded normal `|N(0, sigma_z)|` — a reflecting
  lower border at 0 keeps the DV non-negative.
- **Within-trial noise** is Gaussian (`noise = 0.1`, step `dt = 0.01`).
- The **response offset** is the step at which the bound is crossed relative
  to the target (`stop_k - target`), binned to `-2 … +2`. Trials that never
  stop within range fall into a "no-stop" category.
- Reaction time is the time since the last step onset, scaled to ms.

Offsets are pooled over target numerosities 3 and 4 across both monkeys.

## Files

All scripts are in `script/` and resolve paths relative to their own
location (`data/`, `results/`, `figures/` are created/read as siblings).

| File | Purpose |
|------|---------|
| `diffusion_model_fig.m` | Forward simulation + the manuscript figure set. Runs the model with fixed parameters, loads monkey data, and produces the demo traces, response-frequency, and RT figures. Includes the session-level RT trend/asymmetry tests (Wilcoxon signed-rank). |
| `fit_diffusion_mle_MS.m` | Fits `drift_mean` (and optionally `drift_sigma`), `sigma_z` to monkey **performance** (the multinomial over offset bins + no-stop) by minimising negative log-likelihood. Multi-start `fminsearch` with a sigmoid bound transform. Saves `results/params_perf.mat`. Self-contained (local functions included). |
| `fit_diffusion_rt_MS.m` | Fits the same parameters to monkey **RT distributions** using Ratcliff & Tuerlinckx's G² statistic on RT quantile bins (`[.1 .3 .5 .7 .9]`). Saves `results/params_rt.mat`. |
| `compare_rt_fit_MS.m` | Loads both fitted parameter sets and compares how well each explains the RTs, using G² per offset category and a stratified bootstrap test of `ΔG² = G²(perf) − G²(RT)`. Produces comparison figures. |

## Typical workflow

```matlab
% from the script/ directory in MATLAB
fit_diffusion_mle_MS      % -> results/params_perf.mat
fit_diffusion_rt_MS       % -> results/params_rt.mat
compare_rt_fit_MS         % reads both .mat files, runs the comparison
diffusion_model_ratcliff_MS   % forward model + manuscript figures
```

`diffusion_model_ratcliff_MS.m` uses hard-coded parameters (from the
performance fit) and can be run on its own. The three fitting/comparison
scripts share an identical data-loading and model definition so the fits are
directly comparable.

Figures are written as vector PDFs to `figures/`; fitted parameters to
`results/`. Both directories are created automatically.

## Requirements

- MATLAB with the Statistics and Machine Learning Toolbox
  (`quantile`, `signrank`, `corr`, `fminsearch`).
- **Data** in `data/`: `bhv_data_tbl_m1.mat` and `bhv_data_tbl_m2.mat`,
  each holding a `dataTable` with per-session `RespMat`, `RT_conf`, and
  `RT_err_conf`. In `RespMat`, column 2 is the target numerosity, column 3
  the produced number, and column 5 the outcome (`0` correct, `6` error);
  all-`9` rows are padding and are dropped.
- **Helper functions** `run_diffusion_sim.m` and `compute_g2.m`, used by
  `fit_diffusion_rt_MS.m` and `compare_rt_fit_MS.m`. They must be on the
  MATLAB path for the RT scripts to run.

## Reproducibility

All scripts seed the RNG (`rng(42)`, with fixed seeds for the final
simulations and the bootstrap), so results are reproducible run to run.
