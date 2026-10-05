# Environment Change Detector

The active implementation is the stateless mean/std distance in
`models/supervisor/simple_environment_detector.py`. The older predictive-rank
martingale implementation remains in the other supervisor files for reference
only and should not be extended for the current stage.

This component monitors shifts in frame-level observable features. It does not
estimate BER, select a receiver, or certify that a receiver is safe.

## Inputs and output

`extract_simple_environment_features(rx_dd, h_hat, noise_power, config)` uses
the existing data-domain observation and MMSE receiver to compute an estimated
SNR proxy and an `H_hat` reconstruction-NMSE proxy. `SimpleEnvironmentReference`
stores only training mean/std values. `reference.score(features)` returns one
fresh numeric distance per frame.

The score is stateless: no alarm, threshold, latching, or history is maintained.
It is intended for a future per-frame controller that combines environment
distance with receiver reliability.

## Interpretation

The reconstruction-NMSE value is a live proxy, not the true offline channel
NMSE: true `h_dd` is unavailable in deployment. A low distance only means the
two observable proxies resemble training. It does not prove receiver accuracy.
The simple score also cannot detect a shift that leaves both proxies unchanged.

The current validation split has only 15 frames per regime. Its replay is a
smoke check, not threshold or detection validation. The controlled channel-gain
perturbation is synthetic and not evidence for a physical propagation shift.
Do not inspect the test split to choose a future threshold.

## User-run checks

From the repository root, run the data-free unit tests:

```bash
python -m unittest discover -s tests -p "test_simple_environment_detector.py"
```

Build fixed references from training samples only, reading only `rx_dd` and
`h_hat`:

```bash
python training/supervisor/prepare_simple_environment_detector.py --config configs/experiment_v1.yaml
```

Replay validation samples and run the explicitly synthetic gain diagnostic:

```bash
python evaluations/supervisor/evaluate_simple_environment_detector.py --config configs/experiment_v1.yaml
```

Outputs are written to `experiments/environment_detector_simple/`. No training
or receiver checkpoint evaluation is performed by these commands. The older
martingale test/replay files remain available but are not the active path.