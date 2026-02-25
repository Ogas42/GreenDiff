# gd

Structured rewrite of the original `gf` package in a new parallel package.

## CLI

```bash
python -m gd.cli pipeline run --config gd/configs/default.yaml --profile local_4060_smoke --stages vae green
python -m gd.cli train vae --config gd/configs/default.yaml --profile local_4060 --run-name vae_baseline --seed 42
python -m gd.cli eval green --config gd/configs/default.yaml --profile local_4060 --ckpt-dir gd/runs/<run>/checkpoints --save-json gd/runs/<run>/metrics/green_eval_val.json
python -m gd.cli benchmark run --suite synthetic_main_v1 --config gd/configs/default.yaml --profile local_4060 --mode fast
python -m gd.cli benchmark run --suite synthetic_main_v1 --config gd/configs/default.yaml --profile remote_a6000 --mode full --seeds 3
python -m gd.cli report aggregate --runs-root gd/runs --out gd/runs/green_eval_summary.csv
python -m gd.cli report plots --results gd/runs/green_eval_summary.csv --out-dir gd/runs/figures
```

## Notes

- `gf/` remains the reference baseline.
- `gd/` is the new structured package and pipeline entrypoint.
- `VAETrainer` is migrated to `gd/core/*`; `LatentGreenTrainer` is partially structured and still uses the legacy backend loop.
- Green evaluation supports direct `run(config, ...)` and structured JSON result output for reporting.
- Profiles:
  - `local_4060_smoke`: minimum-cost local smoke runs
  - `local_4060`: local laptop development runs
  - `remote_a6000`: server-side heavier training defaults
