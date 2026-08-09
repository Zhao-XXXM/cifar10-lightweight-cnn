import os
from pathlib import Path

import torch
import torch.nn as nn

from d10_vgg_slim import VGGSlim
from d14_resnet18 import ResNet18
from d17_bottleneck import Bottleneck


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_fp32_size_mb(param_count):
    return param_count * 4 / 1024 / 1024


def conv2d_params(in_channels, out_channels, kernel_size, bias=True):
    weight_params = out_channels * in_channels * kernel_size * kernel_size
    bias_params = out_channels if bias else 0
    return weight_params + bias_params


def linear_params(in_features, out_features, bias=True):
    weight_params = in_features * out_features
    bias_params = out_features if bias else 0
    return weight_params + bias_params


def checkpoint_size_mb(path):
    if not path.exists():
        return None
    return path.stat().st_size / 1024 / 1024


def print_model_summary(name, model, checkpoint_path=None):
    trainable = count_trainable_params(model)
    total = count_all_params(model)
    fp32_mb = estimate_fp32_size_mb(trainable)
    ckpt_mb = checkpoint_size_mb(checkpoint_path) if checkpoint_path else None

    print(f"\n[{name}]")
    print(f"可训练参数量: {trainable:,}")
    print(f"总参数量: {total:,}")
    print(f"FP32 参数估算大小: {fp32_mb:.2f} MB")
    if ckpt_mb is not None:
        print(f"checkpoint 文件大小: {ckpt_mb:.2f} MB")


def print_top_level_breakdown(model):
    print("\n[ResNet18 顶层模块参数量拆解]")
    for name, module in model.named_children():
        params = count_trainable_params(module)
        print(f"{name:10s}: {params:,}")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    checkpoint_dir = root_dir / "checkpoints"
    os.chdir(root_dir)

    vgg_slim = VGGSlim(num_classes=10)
    resnet18 = ResNet18(num_classes=10)
    bottleneck = Bottleneck(in_channels=256, mid_channels=64, stride=2)

    print("Day18: 模型复杂度统计")
    print("=" * 40)

    print("\n[手算公式验证]")
    print(f"Conv2d(3, 32, 3, bias=True): {conv2d_params(3, 32, 3, bias=True):,}")
    print(f"Linear(2048, 256, bias=True): {linear_params(2048, 256, bias=True):,}")
    print(f"BatchNorm2d(32) 可训练参数量: {2 * 32:,}")

    print_model_summary(
        "VGG-Slim",
        vgg_slim,
        checkpoint_dir / "vgg_slim_best.pth",
    )
    print_model_summary(
        "ResNet18",
        resnet18,
        checkpoint_dir / "resnet18_best.pth",
    )
    print_model_summary(
        "Bottleneck 单块",
        bottleneck,
    )

    print_top_level_breakdown(resnet18)
