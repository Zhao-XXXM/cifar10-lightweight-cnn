import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_expected_seeds(text):
    return [int(item.strip()) for item in text.split(",")]


def load_records(path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 Day33 正式结果: {path}\n"
            "请先运行 d33_multiseed.py 的 10 Epoch 三种子实验。"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    if len(records) < 2:
        raise ValueError("至少需要两个训练种子才能计算样本标准差")
    return records


def validate_records(records, expected_seeds):
    seeds = [int(record["training_seed"]) for record in records]
    if sorted(seeds) != sorted(expected_seeds):
        raise ValueError(f"训练种子不完整: expected={expected_seeds}, actual={seeds}")

    widths = {float(record["width_mult"]) for record in records}
    split_hashes = {record["split_sha256"] for record in records}
    test_flags = {record["official_test_evaluated"].strip().lower() for record in records}
    if len(widths) != 1:
        raise ValueError(f"多种子实验混入了不同宽度: {sorted(widths)}")
    if len(split_hashes) != 1:
        raise ValueError("不同种子使用了不同的数据划分")
    if test_flags != {"false"}:
        raise ValueError("Day33 结果不应包含官方测试集评估")


def calculate_statistics(name, values):
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values)
    min_value = min(values)
    max_value = max(values)
    return {
        "metric": name,
        "n": len(values),
        "mean": mean_value,
        "sample_std": std_value,
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
        "cv_percent": 100.0 * std_value / mean_value if mean_value else 0.0,
    }


def save_statistics(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_accuracy_plot(records, stats, path):
    seeds = [int(record["training_seed"]) for record in records]
    values = [float(record["best_val_acc"]) for record in records]
    mean_value = stats["mean"]
    std_value = stats["sample_std"]

    figure, axis = plt.subplots(figsize=(7, 5))
    positions = list(range(len(seeds)))
    axis.scatter(positions, values, s=80, color="#377eb8", label="Single seed")
    axis.errorbar(
        len(seeds),
        mean_value,
        yerr=std_value,
        fmt="o",
        markersize=8,
        capsize=7,
        color="#d95f02",
        label="Mean +/- sample std",
    )
    axis.axhline(mean_value, color="#d95f02", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xticks(positions + [len(seeds)], [str(seed) for seed in seeds] + ["mean"])
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Best validation accuracy (%)")
    axis.set_title("Validation Accuracy Across Training Seeds")
    axis.set_ylim(min(values) - 1.0, max(values) + 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_validation_curves(records, root_dir, path):
    figure, axis = plt.subplots(figsize=(8, 5))
    for record in records:
        history_path = root_dir / record["history_path"]
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        epochs = [item["epoch"] for item in history]
        val_acc = [item["val_acc"] for item in history]
        axis.plot(
            epochs,
            val_acc,
            marker="o",
            linewidth=1.5,
            label=f"seed={record['training_seed']}",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("Validation Curves Across Training Seeds")
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    input_path = root_dir / args.input
    records = load_records(input_path)
    expected_seeds = parse_expected_seeds(args.expected_seeds)
    validate_records(records, expected_seeds)
    records.sort(key=lambda record: int(record["training_seed"]))

    output_dir = root_dir / "checkpoints" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_values = {
        "best_val_acc": [float(record["best_val_acc"]) for record in records],
        "final_val_acc": [float(record["final_val_acc"]) for record in records],
        "best_epoch": [float(record["best_epoch"]) for record in records],
        "train_seconds": [float(record["train_seconds"]) for record in records],
    }
    stats_rows = [
        calculate_statistics(name, values)
        for name, values in metric_values.items()
    ]
    accuracy_stats = stats_rows[0]

    save_statistics(stats_rows, output_dir / "statistics.csv")
    save_accuracy_plot(records, accuracy_stats, output_dir / "accuracy_stability.png")
    save_validation_curves(records, root_dir, output_dir / "validation_curves.png")

    report = {
        "width_mult": float(records[0]["width_mult"]),
        "training_seeds": [int(record["training_seed"]) for record in records],
        "split_sha256": records[0]["split_sha256"],
        "official_test_evaluated": False,
        "statistics": stats_rows,
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Day34: 多随机种子统计分析")
    print(f"seeds={report['training_seeds']}, width_mult={report['width_mult']}")
    print(
        f"Best Val Acc = {accuracy_stats['mean']:.2f}% +/- "
        f"{accuracy_stats['sample_std']:.2f}%"
    )
    print(
        f"min={accuracy_stats['min']:.2f}%, max={accuracy_stats['max']:.2f}%, "
        f"range={accuracy_stats['range']:.2f} percentage points"
    )
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="checkpoints/day33_multiseed/summary.csv",
    )
    parser.add_argument("--expected-seeds", default="42,123,2026")
    parser.add_argument("--output", default="day34_seed_statistics")
    main(parser.parse_args())
