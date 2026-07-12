# Public Demo Data

This folder contains synthetic, public-safe trajectory cases for the evaluator.
It does not contain private NegBioDB rows, real trial records, API outputs, local
paths, or model-generated logs.

Run:

```bash
python examples/run_public_demo.py
```

The demo shows three deployment behaviors:

- a correct supported-evidence trajectory with attribution,
- a rejected invalid-value trajectory,
- a fail-closed out-of-distribution specialist trajectory.

It also includes intentionally bad trajectories so reviewers can see which hard
policy violations are caught.

