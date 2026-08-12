import torch
import torch.nn as nn

from d20_light_cnn import (
    LightVGGBlock,
    LightVGGSlim,
    count_trainable_params,
    estimate_fp32_size_mb,
)


class LightVGGSlimGAP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            LightVGGBlock(3, 32),
            LightVGGBlock(32, 64),
            LightVGGBlock(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def linear_params(in_features, out_features, bias=True):
    weight_params = in_features * out_features
    bias_params = out_features if bias else 0
    return weight_params + bias_params


def print_model_summary(name, model):
    params = count_trainable_params(model)
    size_mb = estimate_fp32_size_mb(params)
    print(f"{name:18s} 参数量: {params:,} | FP32估算大小: {size_mb:.2f} MB")


def print_part_breakdown(name, model):
    print(f"\n[{name} 参数拆解]")
    for part_name, part in model.named_children():
        params = count_trainable_params(part)
        print(f"{part_name:10s}: {params:,}")


def print_shape_flow(model, x):
    print("\n[LightVGGSlim-GAP 形状流]")
    print(f"输入: {x.shape}")

    features = model.features(x)
    print(f"卷积特征: {features.shape}")

    pooled = model.classifier[0](features)
    print(f"GAP 后: {pooled.shape}")

    flattened = model.classifier[1](pooled)
    print(f"Flatten 后: {flattened.shape}")

    logits = model.classifier[2](flattened)
    print(f"Logits: {logits.shape}")


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 32, 32)

    light_vgg = LightVGGSlim(num_classes=10)
    light_gap = LightVGGSlimGAP(num_classes=10)

    print("Day24: Global Average Pooling 轻量分类头")
    print("=" * 50)

    print("\n[分类头手算参数量]")
    print(f"原 Flatten 分类头第一层 Linear(2048, 256): {linear_params(2048, 256):,}")
    print(f"GAP 分类头 Linear(128, 10): {linear_params(128, 10):,}")

    print("\n[模型整体参数量对比]")
    print_model_summary("LightVGG-Slim", light_vgg)
    print_model_summary("LightVGG-Slim-GAP", light_gap)

    old_params = count_trainable_params(light_vgg)
    new_params = count_trainable_params(light_gap)
    print(f"整体参数压缩倍数: {old_params / new_params:.2f}x")

    print_part_breakdown("LightVGG-Slim", light_vgg)
    print_part_breakdown("LightVGG-Slim-GAP", light_gap)

    print_shape_flow(light_gap, dummy_input)
