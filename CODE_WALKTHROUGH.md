# RadioDiff 代码学习与实验记录

## 1. 推荐阅读顺序

1. `configs/radio_sample.yaml`：决定图像尺寸、扩散步数、数据路径、checkpoint 和输出目录。
2. `lib/loaders.py`：读取建筑物、发射机和真实 radio map，并完成归一化。
3. `sample_cond_ldm.py:main`：组装 VAE、条件 U-Net、Latent Diffusion 和 DataLoader。
4. `sample_cond_ldm.py:Sampler`：加载权重、遍历测试集、调用采样器、计算指标和保存图片。
5. `sample_cond_ldm.py:slide_sample`：把 256/320 像素窗口滑过输入场景，再拼接成完整 RM。
6. `denoising_diffusion_pytorch/ddm_const_sde.py`：实现扩散前向过程和反向采样。
7. `denoising_diffusion_pytorch/mask_cond_unet.py`：条件 U-Net，输入场景条件并预测噪声。
8. `denoising_diffusion_pytorch/encoder_decoder.py`：VAE，将 RM 映射到 latent，再解码回图像。

## 2. 一次推理的调用链

```text
radio_sample.yaml
  -> main()
  -> RadioUNet_c(test)
  -> AutoencoderKL + conditional Unet
  -> LatentDiffusion
  -> Sampler.sample()
  -> slide_sample()
  -> model.sample()
  -> PNG + metrics.json
```

## 3. 重点变量

- `sampling_timesteps`：反向扩散步数。越大通常越慢，结果可能更稳定。
- `sample_num`：最多处理多少个测试样本。
- `inference_stop_idx`：测试循环的最后索引；设置为 4 表示处理索引 0 到 4。
- `data.root`：RadioMapSeer 根目录，下面应有 `gain/` 和 `png/`。
- `sampler.ckpt_path`：扩散模型 checkpoint。
- `first_stage.ckpt_path`：独立 VAE checkpoint；若为空，则由主 checkpoint 中的 `first_stage_model` 参数补齐。
- `NMSE/RMSE`：误差指标，越低越好。
- `SSIM/PSNR`：结构和信号质量指标，越高越好。

## 4. 每次实验的操作

修改配置中的 `sampling_timesteps` 和 `save_folder`，然后运行：

```powershell
& "C:\Users\Lenovo\Desktop\RadioDiff-Research-Assessment\.venv\Scripts\python.exe" sample_cond_ldm.py --cfg configs/radio_sample.yaml
```

不要复用已有输出目录。每个实验目录应包含：

```text
生成的 PNG
config_resolved.yaml
run_info.json
metrics.json
```

## 5. 建议学习实验

先运行 `baseline_steps_5`，确认流程；再运行 `baseline_steps_20` 和 `baseline_steps_50`，比较速度与 NMSE/SSIM/PSNR；最后固定 steps=20，把随机种子改为 123，保存到 `seed_123`，分析采样随机性。

## 6. 随机种子修正与结果分析

采样器原始实现会在 `ddm_const_sde.py` 的采样函数内部固定 `torch`、`numpy` 和 `random` 的 seed 为 0，覆盖配置文件中的实验变量。当前版本由 `sampler.seed` 传入 `model_cfg.seed`，采样时读取该值；因此 `seed_123` 与 seed=42 的输出可以真实比较。

使用 `analyze_results.py` 可以不加载模型地复核保存的 PNG：

```powershell
& ..\.venv\Scripts\python.exe RadioDiff_Reproduction\RadioDiff\analyze_results.py `
  --results RadioDiff_Reproduction\results `
  --assets RadioDiff_Reproduction\report_assets
```

脚本生成 `results/per_sample_metrics.csv`，并按逐样本 NMSE 选择较好和较差结果，生成报告所需的输入条件、定性对照和扩散步数对照图。
