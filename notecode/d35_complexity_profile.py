import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from d20_light_cnn import count_trainable_params
from d27_width_multiplier import WidthLightVGGSlimGAP


def parse_widths(text):
    widths = [float(item.strip()) for item in text.split(",")]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("width_mult 必须大于 0")
    return widths


def width_to_name(width):
    return f"width_{str(width).replace('.', '_')}"


def shape_to_text(shape):
    return "x".join(str(value) for value in shape)


def count_layers(model, input_shape):
    layer_rows = []
    hooks = []

    def make_hook(name, module):
        def hook(current_module, inputs, output):
            input_tensor = inputs[0]
            output_tensor = output
            if isinstance(module, nn.Conv2d):
                _, _, output_height, output_width = output_tensor.shape
                kernel_height, kernel_width = module.kernel_size
                input_channels = module.in_channels
                output_channels = module.out_channels
                macs = (
                    output_height
                    * output_width
                    * output_channels
                    * (input_channels // module.groups)
                    * kernel_height
                    * kernel_width
                )
                layer_type = "Conv2d"
                groups = module.groups
            else:
                macs = module.in_features * module.out_features
                layer_type = "Linear"
                groups = 1

            layer_rows.append({
                "layer_name": name,
                "layer_type": layer_type,
                "groups": groups,
                "input_shape": shape_to_text(input_tensor.shape),
                "output_shape": shape_to_text(output_tensor.shape),
                "macs": int(macs),
                "flops": int(2 * macs),
            })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape))

    for hook in hooks:
        hook.remove()
    return layer_rows


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plot(records, path):
    labels = [f"width={record['width_mult']}" for record in records]
    params = [record["params"] / 1000 for record in records]
    macs = [record["macs"] / 1e6 for record in records]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, params, color="#377eb8")
    axes[0].set_title("Trainable Parameters")
    axes[0].set_ylabel("Thousands")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, macs, color="#d95f02")
    axes[1].set_title("MACs per image")
    axes[1].set_ylabel("Million MACs")
    axes[1].tick_params(axis="x", rotation=20)

    for axis, values in zip(axes, (params, macs)):
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.2f}", ha="center", va="bottom")
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Width Multiplier Complexity Comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "checkpoints" / "day35_complexity_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    widths = parse_widths(args.widths)
    all_layer_rows = []
    records = []

    for width in widths:
        model = WidthLightVGGSlimGAP(width_mult=width, num_classes=10)
        layer_rows = count_layers(model, (1, 3, 32, 32))
        for row in layer_rows:
            all_layer_rows.append({"width_mult": width, **row})

        params = count_trainable_params(model)
        macs = sum(row["macs"] for row in layer_rows)
        records.append({
            "width_mult": width,
            "channels": str(model.channels),
            "params": params,
            "params_m": params / 1e6,
            "fp32_size_mb": params * 4 / 1024 / 1024,
            "macs": macs,
            "macs_m": macs / 1e6,
            "flops": 2 * macs,
            "flops_m": 2 * macs / 1e6,
            "input_shape": "1x3x32x32",
        })

    save_csv(records, output_dir / "complexity.csv")
    save_csv(all_layer_rows, output_dir / "layer_details.csv")
    save_comparison_plot(records, output_dir / "complexity_comparison.png")
    (output_dir / "complexity.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Day35: 参数量、MACs 和 FLOPs 统计")
    for record in records:
        print(
            f"width={record['width_mult']} | "
            f"params={record['params']:,} | "
            f"MACs={record['macs_m']:.3f}M | "
            f"FLOPs={record['flops_m']:.3f}M | "
            f"FP32={record['fp32_size_mb']:.3f}MB"
        )
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="0.5,1.0,1.5")
    main(parser.parse_args())
