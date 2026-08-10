import torch
import torch.nn as nn

from d10_vgg_slim import VGGSlim


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.dw_bn(self.depthwise(x)))
        x = self.relu(self.pw_bn(self.pointwise(x)))
        return x


class LightVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class LightVGGSlim(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            LightVGGBlock(3, 32),
            LightVGGBlock(32, 64),
            LightVGGBlock(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_fp32_size_mb(param_count):
    return param_count * 4 / 1024 / 1024


def print_model_summary(name, model):
    params = count_trainable_params(model)
    size_mb = estimate_fp32_size_mb(params)
    print(f"{name:14s} 参数量: {params:,} | FP32估算大小: {size_mb:.2f} MB")


def print_part_breakdown(name, model):
    print(f"\n[{name} 参数拆解]")
    for part_name, part in model.named_children():
        params = count_trainable_params(part)
        print(f"{part_name:10s}: {params:,}")


def print_feature_shapes(model, x):
    print("\n[LightVGGSlim 特征图尺寸变化]")
    print(f"输入: {x.shape}")
    for index, block in enumerate(model.features, start=1):
        x = block(x)
        print(f"Block {index}: {x.shape}")
    return x


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 32, 32)

    baseline = VGGSlim(num_classes=10)
    light_model = LightVGGSlim(num_classes=10)

    print("Day20: Light VGG-Slim 结构验证")
    print("=" * 40)

    print("\n[模型参数量对比]")
    print_model_summary("VGG-Slim", baseline)
    print_model_summary("LightVGG-Slim", light_model)

    baseline_params = count_trainable_params(baseline)
    light_params = count_trainable_params(light_model)
    print(f"参数压缩倍数: {baseline_params / light_params:.2f}x")

    print_part_breakdown("VGG-Slim", baseline)
    print_part_breakdown("LightVGG-Slim", light_model)

    feature_output = print_feature_shapes(light_model, dummy_input)

    logits = light_model(dummy_input)
    print("\n[分类输出验证]")
    print(f"展平特征维度: {feature_output.numel() // feature_output.size(0)}")
    print(f"Logits 输出尺寸: {logits.shape}")
