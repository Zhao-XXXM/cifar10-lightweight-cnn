import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from d27_width_multiplier import WidthLightVGGSlimGAP


def parse_numbers(text, number_type):
    values = [number_type(item.strip()) for item in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError("输入列表中的值必须大于 0")
    return values


def width_to_name(width):
    return f"width_{str(width).replace('.', '_')}"


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA 设备")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, probability):
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def benchmark(model, input_tensor, device, warmup, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_tensor)
        synchronize(device)

        latency_ms = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter_ns()
            model(input_tensor)
            synchronize(device)
            end = time.perf_counter_ns()
            latency_ms.append((end - start) / 1e6)
    return latency_ms


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(records, path):
    widths = sorted({record["width_mult"] for record in records})
    batch_sizes = sorted({record["batch_size"] for record in records})
    colors = ["#377eb8", "#d95f02", "#4daf4a", "#984ea3"]
    x_positions = list(range(len(widths)))
    bar_width = 0.8 / len(batch_sizes)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for batch_index, batch_size in enumerate(batch_sizes):
        selected = [
            next(
                record for record in records
                if record["width_mult"] == width and record["batch_size"] == batch_size
            )
            for width in widths
        ]
        offsets = [
            position - 0.4 + bar_width / 2 + batch_index * bar_width
            for position in x_positions
        ]
        axes[0].bar(
            offsets,
            [record["median_per_image_ms"] for record in selected],
            width=bar_width,
            color=colors[batch_index % len(colors)],
            label=f"batch={batch_size}",
        )
        axes[1].bar(
            offsets,
            [record["throughput_images_s"] for record in selected],
            width=bar_width,
            color=colors[batch_index % len(colors)],
            label=f"batch={batch_size}",
        )

    labels = [str(width) for width in widths]
    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Width multiplier")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[0].set_title("Median latency per image")
    axes[0].set_ylabel("Milliseconds")
    axes[1].set_title("Inference throughput")
    axes[1].set_ylabel("Images per second")
    figure.suptitle("Real Inference Benchmark")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "checkpoints" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    widths = parse_numbers(args.widths, float)
    batch_sizes = parse_numbers(args.batch_sizes, int)
    device = resolve_device(args.device)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    summary_rows = []
    raw_rows = []
    for width in widths:
        model = WidthLightVGGSlimGAP(width_mult=width, num_classes=10).to(device)
        checkpoint_path = (
            root_dir / args.checkpoint_root / width_to_name(width) / "best.pth"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 32, 32, device=device)
            samples = benchmark(
                model,
                input_tensor,
                device,
                args.warmup,
                args.repeats,
            )
            median_batch_ms = statistics.median(samples)
            p95_batch_ms = percentile(samples, 0.95)
            summary_rows.append({
                "width_mult": width,
                "batch_size": batch_size,
                "device": str(device),
                "num_threads": torch.get_num_threads(),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "median_batch_ms": median_batch_ms,
                "p95_batch_ms": p95_batch_ms,
                "median_per_image_ms": median_batch_ms / batch_size,
                "throughput_images_s": 1000.0 * batch_size / median_batch_ms,
            })
            for repeat_index, latency_ms in enumerate(samples, start=1):
                raw_rows.append({
                    "width_mult": width,
                    "batch_size": batch_size,
                    "repeat": repeat_index,
                    "latency_ms": latency_ms,
                })

    save_csv(summary_rows, output_dir / "benchmark_summary.csv")
    save_csv(raw_rows, output_dir / "latency_samples.csv")
    save_plot(summary_rows, output_dir / "benchmark_comparison.png")

    environment = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "checkpoint_root": args.checkpoint_root,
        "timed_scope": "model forward only",
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Day36: 真实推理延迟与吞吐量")
    for record in summary_rows:
        print(
            f"width={record['width_mult']} | batch={record['batch_size']} | "
            f"median={record['median_batch_ms']:.3f}ms/batch | "
            f"p95={record['p95_batch_ms']:.3f}ms/batch | "
            f"throughput={record['throughput_images_s']:.2f} images/s"
        )
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="0.5,1.0,1.5")
    parser.add_argument("--batch-sizes", default="1,64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/day32_proper_width_models",
    )
    parser.add_argument("--output", default="day36_inference_benchmark")
    main(parser.parse_args())
