# 实验结果目录

每个子目录代表一次独立实验，禁止覆盖已有目录。

| 目录 | 用途 |
|---|---|
| `baseline_steps_5` | 基线，5 个扩散步 |
| `baseline_steps_20` | 20 个扩散步对照 |
| `baseline_steps_50` | 50 个扩散步对照 |
| `seed_123` | 改变随机种子的对照 |
| `dpm_car` | 带车辆输入的场景（需匹配 checkpoint） |

运行脚本会自动写入 `config_resolved.yaml`、`run_info.json` 和 `metrics.json`。PNG 文件是生成的 radio map；建议将真实 RM、输入场景和误差图另存为同一目录，使用一致的样本编号。

## 本地已完成结果

当前权重为 `../RadioDiff/model/model-irt.pt`，数据为 `../RadioDiff/data/archive`，每次只推理测试集中的前 5 个样本（样本名为 `289_0` 至 `289_4`）。聚合指标如下，数值越小越好的是 NMSE/RMSE，越大越好的是 SSIM/PSNR。

| 实验 | seed | sampling steps | NMSE | RMSE | SSIM | PSNR |
|---|---:|---:|---:|---:|---:|---:|
| baseline_steps_5 | 42 | 5 | 0.010148 | 0.039451 | 0.942289 | 28.1332 |
| baseline_steps_20 | 42 | 20 | 0.010325 | 0.039745 | 0.937611 | 28.0739 |
| baseline_steps_50 | 42 | 50 | 0.010602 | 0.040294 | 0.936071 | 27.9523 |
| seed_123 | 123 | 5 | 0.010343 | 0.039785 | 0.941220 | 28.0642 |

`per_sample_metrics.csv` 由 `../analyze_results.py` 生成，包含每个样本的 NMSE、RMSE、SSIM 和 PSNR。`../report_assets` 中保存了报告用的定性对比图。
