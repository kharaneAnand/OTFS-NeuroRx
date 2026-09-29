# Environment Change Detector

This component monitors shifts in frame-level observable features. It does not
estimate BER, select a receiver, or certify that a receiver is safe.

## Inputs and output

`extract_environment_features(rx_dd, h_hat)` reads only the received grid and
estimated channel. Its received/channel energy fields use logarithmic power so
multiplicative gain changes are not compressed at low power.
`EnvironmentChangeDetector.update(features, regime_id)`
returns `NO_SHIFT_EVIDENCE`, `SHIFT_DETECTED`, or a pooled-reference status for
an unsupported context. A trusted `regime_id` can be created with
`make_regime_key(snr_db, velocity_kmh)`. If no independent context label is
available, omit it to use the pooled training reference.

`NO_SHIFT_EVIDENCE` is not proof that the environment is unchanged or that a
receiver's error rate is acceptable. A detected alarm is latched in the detector
state and remains active until a new monitor lifecycle is deliberately created.

## Statistical contract

The predictive rank martingale uses fixed training-split references, randomized
tie-breaking, order and dispersion rank features, and ONS betting. The runtime
default allocates alpha by Bonferroni over every configured reference and scalar
feature. Per-context mode allocates alpha over one independently trusted
context's features only; it has no combined false-alarm bound across context
changes.
The anytime marginal false-alarm guarantee requires a fixed feature map and
independent calibration and online frame vectors that are i.i.d. under the
no-shift null. It is not conditional on every realized reference and does not
cover arbitrary serial dependence. Keep each frame as one observation; do not
treat its grid cells as independent samples.

The current validation split has only 15 frames per regime. Its replay is a
smoke check, not empirical validation of a 5% false-alarm rate. The controlled
channel-gain perturbation is a feature-pipeline diagnostic, not evidence for a
physical propagation shift. Do not inspect the test split to choose features or
thresholds.

## User-run checks

From the repository root, run the data-free unit tests:

```bash
python -m unittest discover -s tests -p "test_environment_detector.py"
```

Build fixed references from training samples only, reading only `rx_dd` and
`h_hat`:

```bash
python training/supervisor/prepare_environment_detector.py --config configs/experiment_v1.yaml
```

Replay validation samples and run the explicitly synthetic gain diagnostic:

```bash
python evaluations/supervisor/replay_environment_detector.py --config configs/experiment_v1.yaml
```

Outputs are written to `experiments/environment_detector/`. No training or
receiver checkpoint evaluation is performed by these commands.

References must be rebuilt after a feature-schema change.