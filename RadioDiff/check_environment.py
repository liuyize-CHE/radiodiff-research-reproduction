"""Check core imports and GPU forward/backward without datasets or downloads."""
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".tools" / "matplotlib"))

import torch
import torchvision
from accelerate import Accelerator


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; torchvision: {torchvision.__version__}")
    for name in (
        "train_vae", "train_cond_ldm", "sample_cond_ldm",
        "denoising_diffusion_pytorch.mask_cond_unet",
        "denoising_diffusion_pytorch.ddm_const_sde",
        "pytorch_lightning", "timm",
    ):
        importlib.import_module(name)
        print(f"Import OK: {name}")
    assert torch.cuda.is_available(), "CUDA is unavailable"
    print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA: {torch.version.cuda}")
    accelerator = Accelerator(mixed_precision="no")
    model = torch.nn.Conv2d(3, 8, 3, padding=1).to(accelerator.device)
    x = torch.randn(2, 3, 32, 32, device=accelerator.device)
    loss = model(x).square().mean()
    accelerator.backward(loss)
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(model.weight.grad).all()
    print("PASS: imports, Accelerate, CUDA convolution and backward")


if __name__ == "__main__":
    main()
