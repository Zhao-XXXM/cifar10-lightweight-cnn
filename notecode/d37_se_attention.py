import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from d20_light_cnn import DepthwiseSeparableConv, count_trainable_params
from d27_width_multiplier import WidthLightVGGSlimGAP, make_channels


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.squeeze(x).flatten(1)
        scale = self.excitation(scale).view(x.size(0), self.channels, 1, 1)
        return x * scale


class SEWidthLightVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.se1 = SEBlock(out_channels, reduction)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.se2 = SEBlock(out_channels, reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.se1(self.conv1(x))
        x = self.se2(self.conv2(x))
        return self.pool(x)


class SEWidthLightVGGSlimGAP(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, reduction=16):
        super().__init__()
        c1, c2, c3 = make_channels([32, 64, 128], width_mult)
        self.channels = (c1, c2, c3)
        self.reduction = reduction
        self.features = nn.Sequential(
            SEWidthLightVGGBlock(3, c1, reduction),
            SEWidthLightVGGBlock(c1, c2, reduction),
            SEWidthLightVGGBlock(c2, c3, reduction),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_se_params(model):
    return sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, SEBlock)
        for parameter in module.parameters()
    )


def parse_widths(text):
    widths = [float(item.strip()) for item in text.split(",")]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("width_mult 必须大于 0")
    return widths


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def inspect_width(width, reduction):
    baseline = WidthLightVGGSlimGAP(width_mult=width, num_classes=10)
    se_model = SEWidthLightVGGSlimGAP(
        width_mult=width,
        num_classes=10,
        reduction=reduction,
    )
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        baseline_logits = baseline(dummy_input)
        se_logits = se_model(dummy_input)

    expected_shape = (2, 10)
    if tuple(se_logits.shape) != expected_shape:
        raise ValueError(f"SE 模型输出形状错误: {se_logits.shape}")
    if not torch.isfinite(se_logits).all():
        raise ValueError("SE 模型输出包含 NaN 或 Inf")

    baseline_params = count_trainable_params(baseline)
    se_params = count_se_params(se_model)
    total_params = count_trainable_params(se_model)
    return {
        "width_mult": width,
        "channels": str(se_model.channels),
        "reduction": reduction,
        "baseline_params": baseline_params,
        "se_params": se_params,
        "se_total_params": total_params,
        "added_params": total_params - baseline_params,
        "added_ratio_percent": 100.0 * (total_params - baseline_params) / baseline_params,
        "baseline_output_shape": str(tuple(baseline_logits.shape)),
        "se_output_shape": str(tuple(se_logits.shape)),
    }


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "checkpoints" / "day37_se_structure"
    output_dir.mkdir(parents=True, exist_ok=True)
    widths = parse_widths(args.widths)
    records = [inspect_width(width, args.reduction) for width in widths]

    save_csv(records, output_dir / "se_structure.csv")
    (output_dir / "se_structure.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Day37: SE 通道注意力结构验证")
    for record in records:
        print(
            f"width={record['width_mult']} | "
            f"baseline={record['baseline_params']:,} | "
            f"SE params={record['se_params']:,} | "
            f"SE total={record['se_total_params']:,} | "
            f"added={record['added_ratio_percent']:.2f}%"
        )
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="0.5,1.0,1.5")
    parser.add_argument("--reduction", type=int, default=16)
    main(parser.parse_args())
