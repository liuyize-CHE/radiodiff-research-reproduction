# RadioDiff Research Reproduction

This repository contains the reproducible code, configurations, saved metrics, and visualization assets for the RadioDiff baseline inference experiments.

The repository intentionally does not include the RadioMapSeer dataset, pretrained checkpoints, local Python environments, or cache files. Place the dataset and checkpoint locally before running inference.

## Experiments

| Experiment | Seed | Sampling steps | NMSE | RMSE | SSIM | PSNR |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_steps_5` | 42 | 5 | 0.010148 | 0.039451 | 0.942289 | 28.1332 |
| `baseline_steps_20` | 42 | 20 | 0.010325 | 0.039745 | 0.937611 | 28.0739 |
| `baseline_steps_50` | 42 | 50 | 0.010602 | 0.040294 | 0.936071 | 27.9523 |
| `seed_123` | 123 | 5 | 0.010343 | 0.039785 | 0.941220 | 28.0642 |

Each formal result directory contains five generated radio maps, `config_resolved.yaml`, `run_info.json`, and `metrics.json`. The experiments use the first five samples of the test loader for fast, controlled comparisons.

## Environment

- Python 3.10.21
- PyTorch 2.7.1 with CUDA 12.8
- NVIDIA RTX 5060 Laptop GPU was used for the recorded runs

The original source code and Windows dependency notes are in `RadioDiff/`. The source code is adapted from the official RadioDiff repository and retains its original license.

## Local setup

1. Prepare the RadioMapSeer archive with the directory structure expected by `RadioDiff/lib/loaders.py`.
2. Place the checkpoint at `RadioDiff/model/model-irt.pt`, or change `sampler.ckpt_path` in `RadioDiff/configs/radio_sample.yaml`.
3. Change `data.root` in the same configuration to the local dataset path.
4. Run the environment check:

```powershell
& .\.venv\Scripts\python.exe RadioDiff\check_environment.py
```

5. Run the default five-step inference:

```powershell
& .\.venv\Scripts\python.exe RadioDiff\sample_cond_ldm.py --cfg RadioDiff\configs\radio_sample.yaml
```

6. Analyze saved PNG files without loading the model:

```powershell
& .\.venv\Scripts\python.exe analyze_results.py --results results --assets report_assets
```

## Seed correction

The original sampling function reset the PyTorch, NumPy, and Python random seeds to zero inside the sampler, which made a configuration-level seed comparison ineffective. This repository passes `sampler.seed` through `model_cfg.seed` and reads it inside the sampler. The change is documented in `CODE_WALKTHROUGH.md`.

## Results and report assets

- `results/`: saved generated maps and JSON metrics
- `results/per_sample_metrics.csv`: per-sample NMSE, RMSE, SSIM, and PSNR
- `report_assets/`: input condition, qualitative best/worst cases, and sampling-step comparison figures
- `results_README_source.md`: original result-directory notes
