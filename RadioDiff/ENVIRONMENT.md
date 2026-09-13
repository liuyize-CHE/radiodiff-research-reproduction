# Windows / RTX 5060 环境

环境位于工作区根目录 `.venv`，Python 安装在 `.tools/python`。不要移动这两个目录。
使用 Python 3.10、PyTorch 2.7.1 + CUDA 12.8、torchvision 0.22.1。
原 README 的 torch 1.12 / CUDA 11.3 不适用于本机 RTX 50 系显卡。
CUDA 运行库随 PyTorch wheel 安装，正常运行无需另装 CUDA Toolkit。

2026-09-12 本机验证通过：88 个包无依赖冲突，训练/推理入口和核心模型模块导入成功，
RTX 5060 Laptop GPU 上 CUDA 卷积及反向传播成功，Accelerate 单进程启动验证成功。
完整训练和真实权重推理尚未执行。旧版 Accelerate 会输出 `pkg_resources` 弃用提示，不影响本次验证。

## 使用

从工作区根目录打开 PowerShell：

```powershell
& .\.venv\Scripts\Activate.ps1
cd RadioDiff_Reproduction\RadioDiff
python check_environment.py
```

如果 PowerShell 禁止执行激活脚本，可以直接调用解释器，无需更改系统执行策略：

```powershell
cd RadioDiff_Reproduction\RadioDiff
& ..\..\.venv\Scripts\python.exe check_environment.py
& ..\..\.venv\Scripts\python.exe train_vae.py --help
```

VS Code 已配置工作区 `.venv/Scripts/python.exe`。

## 安装与复现

在工作区根目录执行（需要联网；uv 已下载到 `.tools`）：

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.tools\python"
$env:UV_CACHE_DIR = "$PWD\.tools\cache"
& .\.tools\uv.exe venv --python 3.10 .venv
& .\.tools\uv.exe pip install --python .venv/Scripts/python.exe torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
& .\.tools\uv.exe pip install --python .venv/Scripts/python.exe -r RadioDiff_Reproduction/RadioDiff/requirements-windows.txt
& .\.tools\uv.exe pip check --python .venv/Scripts/python.exe
```

`requirements-windows.txt` 保留核心旧版依赖，并补全 pandas、matplotlib、PyWavelets、torch-pruning 等入口所需依赖。
`requirements-windows.lock.txt` 记录本次安装的全部确切版本；若需复现这些版本，先按上面的 CUDA 源安装 PyTorch，
再将其他依赖安装命令中的 `requirements-windows.txt` 换为 `requirements-windows.lock.txt`。
不要在此环境安装 `TFMQ/requirements.txt`：它属于另一个实验，并指向 CUDA 12.1 nightly 源。

## 训练前还需配置

环境验证不等于完成训练或推理。当前数据位于 `data/archive`，已有模型 `model/model-irt.pt`；
尚未验证数据完整性及该权重与目标配置的匹配关系。

- `lib/loaders.py` 的数据集默认路径仍含原作者的 `/home/.../RadioMapSeer/`，运行目标任务前应改为实际数据路径，或在构造数据集时传入 `dir_dataset`。
- `configs/first_radio.yaml`、`configs/radio_train.yaml`、`configs/radio_sample.yaml` 的输出和权重路径仍需按目标实验设置。
- `radio_train.yaml` 原始 batch_size 为 66，8 GB 显卡需从较小值（例如 1）开始，结合实际显存调整；完整模型是否能训练需实际验证。
- 首次构建模型可能下载 ImageNet/LPIPS 权重。

调整好实验配置后，在源码目录启动单 GPU 训练：

```powershell
& ..\..\.venv\Scripts\accelerate.exe launch --config_file configs/accelerate-local.yaml train_vae.py --cfg configs/first_radio.yaml
& ..\..\.venv\Scripts\accelerate.exe launch --config_file configs/accelerate-local.yaml train_cond_ldm.py --cfg configs/radio_train.yaml
```

兼容性修改：三个骨干网络文件中的私有 `_ModelURLs` 替换为普通字典；
删除 `train_cond_ldm.py` 导入时向原作者 Linux 主目录写 Accelerate 配置的行为，改用显式本地配置。

参考：https://pytorch.org/blog/pytorch-2-7/ （Blackwell / CUDA 12.8 支持）
