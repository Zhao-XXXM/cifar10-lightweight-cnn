import argparse

import torch
import torch.nn as nn

from d20_light_cnn import DepthwiseSeparableConv, count_trainable_params


def make_channels(base_channels, width_mult):
    return [max(1, int(channels * width_mult)) for channels in base_channels]


class WidthLightVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class WidthLightVGGSlimGAP(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        c1, c2, c3 = make_channels([32, 64, 128], width_mult)
        self.channels = (c1, c2, c3)

        self.features = nn.Sequential(
            WidthLightVGGBlock(3, c1),
            WidthLightVGGBlock(c1, c2),
            WidthLightVGGBlock(c2, c3),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def estimate_fp32_size_mb(param_count):
    return param_count * 4 / 1024 / 1024


def print_model_summary(width_mult):
    model = WidthLightVGGSlimGAP(width_mult=width_mult)
    params = count_trainable_params(model)
    size_mb = estimate_fp32_size_mb(params)

    dummy_input = torch.randn(2, 3, 32, 32)
    features = model.features(dummy_input)
    logits = model(dummy_input)

    print(f"\n[width_mult={width_mult}]")
    print(f"通道数: {model.channels}")
    print(f"参数量: {params:,}")
    print(f"FP32估算大小: {size_mb:.2f} MB")
    print(f"卷积特征尺寸: {features.shape}")
    print(f"Logits尺寸: {logits.shape}")
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--widths",
        default="0.5,1.0,1.5",
        help="逗号分隔的 width multiplier",
    )
    args = parser.parse_args()

    widths = [float(item.strip()) for item in args.widths.split(",")]

    print("Day27: Width Multiplier 结构验证")
    print("=" * 44)

    baseline_params = None
    for width in widths:
        params = print_model_summary(width)
        if width == 1.0:
            baseline_params = params

    if baseline_params is not None:
        print("\n[相对 width_mult=1.0 的参数比例]")
        for width in widths:
            params = count_trainable_params(
                WidthLightVGGSlimGAP(width_mult=width)
            )
            print(f"width_mult={width}: {params / baseline_params:.3f}x")


if __name__ == "__main__":
    main()
