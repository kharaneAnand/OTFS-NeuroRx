# OTFS-NeuroRx

Research code for generating an OTFS dataset, developing neural and model-based receivers, evaluating them on a fixed test split, and prototyping an environment-change monitor for a future adaptive receiver bank.

## Project Status

- **Best evaluated receiver:** OAMP-DL, with 20 learned iterations.
- **Current bank default for a fresh decision pipeline:** OAMP-DL. The intent is for a selected/last-used model to become the next default; automatic switching is not implemented yet.
- **GNN receiver:** the MMSE-initialized GNN remains the original GNN candidate/third receiver. The neutral-initialized GNN is an initialization ablation.
- **PI-EGNN:** retained as an experimental candidate, but its run is a documented negative and it did not meet the pre-registered bar to replace the original GNN.
- **Environment-change detector:** the active implementation is now a stateless mean/std distance score. The older martingale implementation is preserved for reference but is not the active path. The score is not connected to receiver selection yet.
- **Few-shot OAMP-DL adaptation:** an offline, manually invoked experimental script now adapts only OAMP-DL's 60 scalar controls for the logged 10 dB / 500 km/h ADAPT condition. Automatic controller-triggered adaptation or checkpoint promotion is not implemented.

## Dataset and Fairness

The V1 dataset has 900 frames: 630 training, 135 validation, and 135 test frames. It is stratified across nine conditions: SNR {10, 15, 20} dB and velocity {30, 120, 500} km/h, with 100 generated frames per condition and 15 test frames per condition. The configured random seed is 42.

Each NPZ sample contains `rx_dd`, `tx_dd`, `h_dd`, `h_hat`, and channel-path metadata. `h_dd` is the simulated true channel and is **not an allowed receiver input**. Receivers and the environment detector use the received frame and pilot-derived `h_hat`; `tx_dd` is used only as a training target or for offline metric calculation. The detector reads only `rx_dd` and `h_hat`.

Key paths:

- Raw samples: `datasets/otfs/raw/sample_*.npz`
- Split metadata: `datasets/otfs/processed/dataset_v1_split.csv`
- Source metadata: `datasets/otfs/metadata/dataset_v1_metadata.csv`
- Main experiment configuration: `configs/experiment_v1.yaml`

The configuration describes a 16 x 32 received delay-Doppler grid, a 432 x 368 data-domain estimated channel, and 368 QPSK data symbols per frame. Current evaluation reports per-frame BER, SER, and NMSE; aggregate summaries generally use mean +/- 1.96 standard errors for CI95. Small per-condition test groups (15 frames) mean these intervals should be interpreted cautiously.

## Unified Pipeline Application

The root `run_pipeline.py` runs a single frame or a dataset split through the active receiver adapter, environment-distance extraction, reliability-reference scoring, and the current stateful controller. It starts on OAMP-DL unless a prior controller state is provided. The controller's existing lazy candidate inference, confidence thresholds, cooldown, and frozen BER gate are unchanged. Environment distance is reported but does not affect the controller action.

Run one existing frame (the SNR and velocity are required because they are not stored inside the NPZ):

```bash
python run_pipeline.py --input datasets/otfs/raw/sample_000000.npz --snr 10 --velocity 30
```

Generate one frame in the run's scratch directory with a seed different from V1:

```bash
python run_pipeline.py --generate --snr 10 --velocity 500 --seed 43
```

Run the current test split, or generate a fully independent 900-frame dataset and run its test split:

```bash
python run_pipeline.py --dataset-root datasets/otfs --split test
python run_pipeline.py --generate-dataset --seed 43 --output-dir experiments/generalization_seed43
```

`--generate-dataset` writes the new dataset and split below the new run directory; it does not overwrite `datasets/otfs/`. In this mode the app performs fresh inference for each receiver on each evaluated frame and computes BER/SER/NMSE directly from that frame's `tx_dd` after routing. It does not use the original test-index metric CSVs. Thus it resolves the prior fresh-per-frame-scoring gap for this batch workflow; the trained checkpoints and reference statistics remain fixed from the original training data.

The JSON frame output contains all complex symbol estimates as real/imaginary pairs and QPSK hard bits. The bit rule is `real >= 0` gives bit 0 value `1` (otherwise `0`), and `imag >= 0` gives bit 1 value `1` (otherwise `0`). Metrics are marked `reference_only_requires_ground_truth` when `tx_dd` exists; without `tx_dd`, BER/SER/NMSE are all `null` and the status is `unavailable_no_ground_truth`.

Batch output includes `summary.json`, `frames.jsonl`, `decision_log.csv`, and `predictions.npz`. Supply `--state-out state.json` and later `--state-in state.json --state-out next_state.json` to carry active receiver, low-confidence streak, switch cooldown, and frame counter between invocations. Outputs go to a new timestamped folder under `experiments/pipeline_runs/` unless `--output-dir` is supplied; existing output directories are refused.

An OAMP-DL `ADAPT` recommendation is run against the matching 15-frame validation condition when a calibration dataset is available, using the existing fixed-step few-shot settings and a separate run-local checkpoint. This does not promote or overwrite the base checkpoint. Other receiver recommendations are reported as unsupported by the existing OAMP-DL-only adaptation routine. A single frame without a calibration dataset can only report the recommendation.

## Receiver Models

| Model | Description | Current evidence/status |
|---|---|---|
| MMSE with `H_hat` | Linear data-domain reference detector. | Baseline: BER 0.06688, SER 0.12764, NMSE 0.36825. |
| Blind DNN | Fully connected receiver using `rx_dd` without the explicit channel estimate. | Documented negative/legacy result. Its saved summary reports BER 0.74899 alongside SER 0.00193, which is internally inconsistent; do not use that artifact for ranking without correcting and re-evaluating it. |
| MMSE-residual DNN | Learns a correction to the MMSE estimate. | Negative: did not improve BER, SER, and NMSE together. |
| Informed MMSE-residual DNN | Residual receiver with channel/residual side features. | Negative: did not improve BER, SER, and NMSE together. |
| Plain analytical OAMP | Fixed analytical QPSK OAMP detector, evaluated with 20 iterations. | Negative against MMSE overall: BER 0.07474, SER 0.14124, NMSE 0.23160. |
| OAMP-DL | Unfolded receiver with learned iteration controls; configured for 20 iterations. | Current best: BER 0.03296, SER 0.06345, NMSE 0.10175; wins all nine conditions against MMSE in the saved evaluation. Fresh pipeline default. |
| Bipartite GNN, MMSE-init | Four-layer graph receiver initialized with the MMSE symbol estimate; graph edges use the fixed 1% of max `abs(H_hat)` threshold. | Original GNN/bank candidate: BER 0.03687, SER 0.07073, NMSE 0.11199. It beats MMSE overall but loses to OAMP-DL overall. |
| Bipartite GNN, neutral-init | Same GNN with neutral/zero symbol initialization. | Initialization ablation: BER 0.04204, SER 0.08084, NMSE 0.12732; worse than MMSE-init GNN and OAMP-DL overall. |
| PI-EGNN | MMSE-initialized GNN enhanced with a 3-step analytical OAMP prior and learned attention over active graph edges. | Experimental candidate; did not beat the original GNN on all three overall metrics and did not beat OAMP-DL overall or in either registered regime. It does not replace the original GNN. |

All values above are means over the 135-sample test split, unless the status text says otherwise. Lower is better. See the linked result directories below for confidence intervals and per-condition data.

## PI-EGNN Experiment

The prior iteration count is locked at 3 in `experiments/pi_egnn/prior_config.json`. A validation-only 3/5/10 check found statistically indistinguishable results and no observed accuracy benefit beyond 3; the choice is for simplicity, not a claimed speed tradeoff. The prior uses `rx_dd`, `H_hat`, and configured per-sample noise power only.

Before its GPU run, the PI-EGNN smoke checks confirmed a finite 368-symbol complex output, active-neighborhood attention sums of one, MMSE-valued initialization, cached-feature agreement on five examples, and no `h_dd` input reference. The transferred GPU artifacts are under `gpu_results/pi_egnn/` in this checkout. This is a copied run bundle; active code paths write under `experiments/pi_egnn/`.

| Receiver | BER | SER | NMSE |
|---|---:|---:|---:|
| PI-EGNN | 0.04003 | 0.07723 | 0.12233 |
| Original MMSE-init GNN | 0.03687 | 0.07073 | 0.11199 |
| OAMP-DL | 0.03296 | 0.06345 | 0.10175 |

Both registered PI-EGNN hypotheses failed: low SNR (10 dB across the three velocities), and 120 km/h across all three SNRs. Per-condition receiver and paired-difference confidence intervals are in `gpu_results/pi_egnn/per_condition_results.csv`; overall results are in `gpu_results/pi_egnn/evaluation_results.json`. Training history, checkpoint, per-sample results, prior sanity check, and regime definition are in the same folder.

Measured PI-EGNN end-to-end estimate was about 0.0762 seconds per sample, including feature extraction and input preparation, compared with about 0.00574 seconds per sample for OAMP-DL. The cache reduced repeated work during training/evaluation, but the production analytical feature builder still uses NumPy MMSE/OAMP calculations. A separate solver benchmark found Torch CUDA solve faster for the tested case; the pipeline was not switched to that implementation.

## Environment-Change Detector

The active detector is a standalone, target-free, **stateless distance scorer**. It is not the reliability estimator, does not predict BER, does not select/replace the default receiver, and does not trigger adaptation.

For each frame it reads only `rx_dd`, `H_hat`, configured noise power, and the existing OTFS configuration. It computes two cheap features:

1. `estimated_snr_db`: $10\log_{10}(||y||^2 / ||y - H_hat x_MMSE||^2)$, where `y` is the existing data-domain observation and `x_MMSE` is the existing MMSE estimate.
2. `h_hat_residual_nmse_proxy`: $||y - H_hat x_MMSE||^2 / ||y||^2$.

The second value is an inference-time reconstruction-error proxy. It is **not** the true `H_hat` NMSE used in offline channel-estimation evaluation, because true `h_dd` is unavailable in a live decision loop.

The reference builder computes the mean and sample standard deviation of these two features from the 630 training samples. Every incoming frame is scored independently:

$$
D(x) = \sqrt{\frac{1}{2}\sum_i\left(\frac{x_i-\mu_i}{\sigma_i}\right)^2}.
$$

The detector returns this numeric score only. It has no threshold, alarm, memory, latching, sequential betting, or context-specific calibration. A future controller can call it repeatedly and combine it with reliability estimates for KEEP/SWITCH/ADAPT logic.

The previous predictive-rank-martingale implementation remains in `models/supervisor/environment_features.py`, `models/supervisor/environment_change.py`, and the original `training/supervisor/prepare_environment_detector.py` / `evaluations/supervisor/replay_environment_detector.py` files for reference. The new active path is:

- Simple features and stateless reference: `models/supervisor/simple_environment_detector.py`
- Reference builder: `training/supervisor/prepare_simple_environment_detector.py`
- Validation evaluator: `evaluations/supervisor/evaluate_simple_environment_detector.py`
- Unit tests: `tests/test_simple_environment_detector.py`
- Outputs: `experiments/environment_detector_simple/validation_results.json` and `validation_condition_scores.csv`

The simple validation is independent of calibration: the reference uses training frames, while validation frames receive both a nominal score and a score after independent additive complex corruption of `H_hat`. It also reports nominal mean score for all nine real validation conditions. Both features use the same reconstruction residual; the estimated-SNR feature additionally normalizes by received signal power, so this is informative but not a fully independent two-signal validation. The perturbation check is diagnostic, not a physical-channel benchmark. The score is not guaranteed to increase for every possible environmental change, and no thresholding claim is made yet.

Latest real-condition validation results (15 validation frames per cell):

| SNR (dB) | Velocity (km/h) | Mean distance | Median distance |
|---:|---:|---:|---:|
| 10 | 30 | 0.578 | 0.405 |
| 10 | 120 | 1.021 | 0.739 |
| 10 | 500 | 0.689 | 0.502 |
| 15 | 30 | 0.591 | 0.362 |
| 15 | 120 | 0.548 | 0.512 |
| 15 | 500 | 0.601 | 0.537 |
| 20 | 30 | 0.781 | 0.844 |
| 20 | 120 | 0.459 | 0.386 |
| 20 | 500 | 0.733 | 0.645 |

The highest mean distance is 10 dB / 120 km/h; the lowest is 20 dB / 120 km/h. The lack of a simple monotonic SNR/velocity pattern is expected for a pooled two-feature distance: it reports how unusual the observable proxies are relative to all training frames, not physical severity or receiver error. The complete table is saved in `experiments/environment_detector_simple/validation_condition_scores.csv`.

## Setup

There is currently no root `requirements.txt` or `pyproject.toml`. Use a virtual environment and install dependencies into that environment. For the core receiver pipeline, imports include NumPy, pandas, PyYAML, PyTorch, and `textremo-toolbox` (the bundled `Phy_Mod_OTFS/OTFSResGrid.py` imports `textremo_toolbox`). PyTorch installation must match the machine's desired CPU/CUDA environment. Some unrelated plots and bundled toolbox tests additionally use Matplotlib and SciPy. `whatshow-toolbox` is not needed by the core receiver path; importing it may require TensorFlow due to its package-level imports.

Example Linux setup for a CUDA 12.8 PyTorch wheel (select a PyTorch build appropriate for your system if different):

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas PyYAML textremo-toolbox
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check the core imports from the repository root:

```bash
python -c "import numpy, pandas, yaml, torch, textremo_toolbox; from src.receivers.mmse import build_data_observation; print('Core imports OK')"
```

## Data Pipeline

The generation and preprocessing scripts use `configs/experiment_v1.yaml` internally and currently have no command-line arguments. Run them from the repository root when creating a fresh dataset:

```bash
python experiments/dataset_generation/generate_otfs_dataset.py
python experiments/dataset_generation/validate_otfs_dataset.py
python experiments/dataset_preprocessing/preprocess_otfs_dataset.py
```

Generation produces 900 frames over 3 SNRs x 3 velocities x 100 frames. Preprocessing validates and creates the fixed stratified 630/135/135 split. These scripts can overwrite/generated dataset artifacts; do not rerun them if you intend to preserve the existing dataset without a backup.

## Train and Evaluate

Run commands from the repository root. Training uses the fixed train/validation split and saves checkpoints/history in its configured experiment directory. Evaluation uses the fixed test split.

```bash
# Blind DNN
python training/dnn/train.py --config configs/experiment_v1.yaml
python evaluations/dnn/evaluate.py --config configs/experiment_v1.yaml

# MMSE residual DNN
python training/dnn/train_residual.py --config configs/experiment_v1.yaml
python evaluations/dnn/evaluate_residual.py --config configs/experiment_v1.yaml

# Informed MMSE residual DNN
python training/dnn/train_informed_residual.py --config configs/experiment_v1.yaml
python evaluations/dnn/evaluate_informed_residual.py --config configs/experiment_v1.yaml

# Original GNN: choose one initialization
python training/gnn/train.py --config configs/experiment_v1.yaml --initialization mmse
python training/gnn/train.py --config configs/experiment_v1.yaml --initialization neutral_zero
python evaluations/gnn/evaluate.py --config configs/experiment_v1.yaml

# OAMP-DL
python training/oamp/train.py --config configs/experiment_v1.yaml
python evaluations/oamp/evaluate.py --config configs/experiment_v1.yaml
```

The analytical MMSE and plain OAMP baseline scripts are fixed-config experiments under `experiments/baseline/`; their exact output artifacts are under `experiments/baseline/` and `experiments/oamp/`. The root config also defines seed 42, model training settings, fixed GNN edge threshold, and output paths.

### PI-EGNN (experimental; not a bank-slot promotion)

The prior sanity check and cache/smoke check do not train. After reviewing them, training and test evaluation are separate commands:

```bash
# Optional: inspect 3/5/10 analytical-prior iterations on validation data
python training/gnn/train_pi_egnn.py --config configs/experiment_v1.yaml --prior-sanity-only

# Optional pre-training cache, shape, attention, leakage and solver checks
python training/gnn/check_pi_egnn.py --config configs/experiment_v1.yaml --benchmark-solvers

# Training and test evaluation
python training/gnn/train_pi_egnn.py --config configs/experiment_v1.yaml
python evaluations/gnn/evaluate_pi_egnn.py --config configs/experiment_v1.yaml
```

The checker and replay commands do not train. PI-EGNN outputs are written under `experiments/pi_egnn/` when run in this checkout. Keep all historical result folders intact when comparing experiments.

### Environment Detector (standalone prototype)

Build the simple mean/std reference, run its unit tests, and evaluate nominal versus perturbed validation frames:

```bash
python -m unittest discover -s tests -p "test_simple_environment_detector.py"
python training/supervisor/prepare_simple_environment_detector.py --config configs/experiment_v1.yaml
python evaluations/supervisor/evaluate_simple_environment_detector.py --config configs/experiment_v1.yaml
```

The simple builder reads `rx_dd`/`h_hat` only. Rebuild the reference whenever the simple feature schema changes. The perturbed validation scores use synthetic additive `H_hat` corruption and are diagnostic data only. Do not use the score for receiver routing until a later controller defines and validates thresholds using independent data.

## Reliability Detector

The reliability detector is a separate stateless score for each `(frame, receiver)` pair. It is not the controller and does not implement KEEP/SWITCH/ADAPT logic.

For each receiver output it computes two live features:

- `receiver_qpsk_margin`: mean `min(abs(real), abs(imag))` over the receiver's 368 complex QPSK estimates. This is the mean distance to a QPSK decision boundary.
- `environment_distance`: the existing stateless simple environment score.

Each receiver has its own train-split mean/std reference for these two features. The live reliability distance is:

$$
R_r = \\sqrt{\\frac{1}{2}\\sum_i\\left(\\frac{x_{r,i}-\\mu_{r,i}}{\\sigma_{r,i}}\\right)^2}.
$$

Lower distance means the receiver's current behavior is more typical of its own reference. It is not a probability of correctness and has no threshold or persistent state.

The reference builder runs the frozen receivers freshly on train and validation frames. MMSE is recomputed analytically; OAMP-DL, original GNN, and PI-EGNN load their existing checkpoints. No retraining is performed. Existing per-sample CSVs are test-split artifacts and are not used to build the references. Validation BER/SER/NMSE are stored only for the offline correlation audit.

Run the reference generation and offline validation audit:

```bash
python training/supervisor/prepare_reliability_references.py --config configs/experiment_v1.yaml
python evaluations/supervisor/evaluate_reliability_detector.py
```

Outputs are written to `experiments/reliability_detector/`. The audit reports, separately for MMSE, OAMP-DL, original GNN, and PI-EGNN, confidence-vs-BER correlations and reliability-distance-vs-BER/SER/NMSE correlations. Weak or near-zero correlations are valid findings: they mean the two-feature signal is not predictive enough for that receiver, not that the result should be explained away.

Latest validation audit: all four frozen receivers were run freshly on 630 training and 135 validation frames on CPU; no retraining occurred. The receiver-confidence Spearman correlation with BER was MMSE `-0.509`, OAMP-DL `-0.927`, original GNN `-0.956`, and PI-EGNN `-0.948`, which is a strong and correctly directed signal, especially for the learned receivers. The combined reliability-distance Spearman correlation with BER was MMSE `0.112`, OAMP-DL `-0.249`, original GNN `-0.190`, and PI-EGNN `-0.217`; therefore the combined two-feature distance is not yet a trustworthy monotonic reliability score. Its Pearson correlations were positive (0.334, 0.589, 0.614, and 0.571 respectively), but the disagreement with Spearman shows that outliers and nonlinear behavior matter. This is an important negative finding: the QPSK confidence feature is useful, while adding the environment distance naively can weaken rank ordering. Results are in `experiments/reliability_detector/validation_results.json` and `validation_scores.csv`.

MMSE's confidence-BER relationship is meaningfully weaker than the learned receivers'. MMSE is a fixed linear estimator and was not trained to produce calibrated confidence, while the learned receivers' training process implicitly ties confident outputs to correct outputs. The future controller should therefore interpret or weight MMSE's confidence score differently rather than assuming all four receiver confidence signals are equally trustworthy.

## Rule-Based Controller

The first controller keeps one `active_model` state. It starts at `oamp_dl`, uses only receiver confidence margins for decisions, and logs environment distance separately without using it in the policy. The candidate pool is `mmse`, `oamp_dl`, `original_gnn`, and `pi_egnn`.

Thresholds are locked from training confidence references only:

- Each margin is standardized using that receiver's own training-reference mean/std: `z = (margin - receiver_train_mean) / receiver_train_std`.
- Active confidence is unusually low when its z-score is below `-1.0`.
- A switch candidate is clearly better when its z-score is at least `0.5` above the active model's z-score. This threshold is fixed before evaluating the test set; it is not adjusted to force or suppress a particular receiver switch.
- A candidate must also pass a frozen absolute BER-quality gate: its previously validated aggregate BER must be within `0.005` of both the active model and OAMP-DL's benchmark BER. The fixed source values are in `configs/reliability_controller_v1.json`; PI-EGNN's 0.04003 BER is outside the OAMP-DL cap of 0.03796, while the original GNN's 0.03687 remains inside it. These are prior receiver-bank results, not labels from the controller audit.
- Low confidence must persist for 2 consecutive frames before candidates are evaluated.
- At least 10 frames must pass between switches.

The controller emits `KEEP`, `SWITCH`, or an `ADAPT` recommendation. `ADAPT` identifies the highest-confidence candidate but does not retrain it. The test audit uses live confidence margins for decisions and only afterward joins the existing per-frame BER/SER/NMSE for reporting.

Run the sequential fixed-test audit after reliability references exist:

```bash
python evaluations/supervisor/evaluate_reliability_controller.py --config configs/experiment_v1.yaml
```

The evaluator computes the active receiver first and calls other receiver adapters only after the two-frame low-confidence guard. It also compares `decision` and active-model sequences against an existing `decision_log.csv` when present, reporting `decision_outcomes_unchanged_vs_previous_log`. Outputs are written to `experiments/reliability_controller/evaluation_results.json`, `decision_log.csv`, and `selected_frame_results.csv`; inference call counts are included. The headline comparison is controller-selected BER/SER/NMSE versus always using OAMP-DL, plus frame-level match rate to the receiver with the lowest offline BER. Test labels are loaded only after sequential decisions for this audit and never enter the decision function.

## Few-Shot OAMP-DL Adaptation

`training/supervisor/adapt_oamp_dl_fewshot.py` implements a manual experiment for the two saved controller ADAPT events, both at 10 dB / 500 km/h and both recommending OAMP-DL. It uses 15 labeled validation frames from that condition, updates only the existing 60 OAMP-DL scalar controls for 20 fixed AdamW steps, and applies an anchor penalty. It writes a separate adapted checkpoint; the original OAMP-DL checkpoint is preserved. The same-condition test set is used only afterward for paired comparison.

Latest run on 15 held-out test frames:

| Model | BER (95% CI half-width) | SER (95% CI half-width) | NMSE (95% CI half-width) |
|---|---:|---:|---:|
| Original OAMP-DL | 0.04656 (0.01760) | 0.09022 (0.03360) | 0.14232 (0.05437) |
| Adapted OAMP-DL | 0.04629 (0.01744) | 0.08967 (0.03327) | 0.14241 (0.05446) |

Paired adapted-minus-original differences were BER `-0.000272` (CI95 `[-0.000657, 0.000114]`), SER `-0.000543` (`[-0.001314, 0.000227]`), and NMSE `+0.000094` (`[-0.000041, 0.000229]`). Every paired interval includes zero. This is a single small-sample experiment and is **suggestive, not definitive**: BER/SER improved slightly, NMSE worsened slightly, and there is no clear evidence of a real benefit. The controller-selected condition came from test-time unlabeled ADAPT events; the event frames were not used for fine-tuning, but this remains an exploratory condition-specific result rather than a pristine confirmatory study.

## Result Locations

- DNN: `experiments/dnn/`
- Residual DNNs: `experiments/dnn_residual/`, `experiments/dnn_informed_residual/`
- GNN initializations: `experiments/gnn_mmse_init/`, `experiments/gnn_neutral_init/`
- OAMP-DL: `experiments/oamp_dl/`
- Plain OAMP: `experiments/oamp/`
- PI-EGNN transferred GPU run: `gpu_results/pi_egnn/`
- Few-shot adapted OAMP-DL: `experiments/oamp_dl_adapt_snr10_v500/`
- Active simple detector outputs: `experiments/environment_detector_simple/`
- Preserved martingale detector outputs: `experiments/environment_detector/`
- Classical baselines: `experiments/baseline/`

Typical model result folders contain `best_model.pt`, `training_history.csv`, `evaluation_results.json`, and per-sample/per-condition CSVs. Do not treat a checkpoint alone as validation; compare its evaluation artifact against the same fixed test split and baselines.

## Repository Map

- `configs/`: experiment configuration.
- `datasets/otfs/`: generated NPZ samples, source metadata, and split metadata.
- `Phy_Mod_OTFS/`: bundled OTFS modem/resource-grid implementation.
- `src/receivers/`: observation extraction and classical MMSE/OAMP receiver utilities.
- `models/`: DNN, GNN, OAMP-DL, and supervisor components.
- `training/`: model training, shared trainer, datasets, PI-EGNN cache, detector reference builder.
- `evaluations/`: fixed test-set model evaluation and detector replay.
- `experiments/`: saved results, checkpoints, baselines, and detector artifacts.
- `gpu_results/`: copied PI-EGNN run artifacts transferred from the GPU host; not the configured output directory for future runs.

## Important Limitations

- The 135-sample test set has been used for the reported receiver comparisons; it should not be reused to tune new receiver or detector features/thresholds and then described as an untouched test.
- Per-condition CIs based on 15 frames are noisy.
- The detector identifies changes in its monitored observable features, not harmful receiver-error changes. No alarm is not proof of safety.
- The detector's current controlled-shift replay did not alarm, so it is not ready for operational model selection.
- OAMP-DL is the measured performance leader, but model-default state transfer, reliability-based switching, and efficient adaptation/retraining remain future work.
