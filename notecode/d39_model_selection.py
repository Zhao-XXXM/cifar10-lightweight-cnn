import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_SPLIT_SHA256 = (
    "6242d545f1d70bbd004ba26cd92784461728e0ffb0a64a1f27d1a6421039967e"
)
EXPECTED_SEEDS = {42, 123, 2026}


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(rows, path):
    if not rows:
        raise ValueError(f"Cannot save empty rows: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def require_false_test_flag(record):
    if record["official_test_evaluated"].strip().lower() != "false":
        raise ValueError("A source result evaluated the official Test Set")


def validate_split(record):
    if record["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Source result uses a different data split")


def parse_day33(records):
    if len(records) != 3:
        raise ValueError("Day33 must contain exactly three seeds")
    seeds = {int(record["training_seed"]) for record in records}
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected Day33 seeds: {sorted(seeds)}")
    for record in records:
        validate_split(record)
        require_false_test_flag(record)
        if float(record["width_mult"]) != 1.0:
            raise ValueError("Day33 baseline must use width_mult=1.0")
    return records


def make_baseline_records(day32, day33, complexity, latency):
    day33 = parse_day33(day33)
    complexity_by_width = {
        float(row["width_mult"]): row for row in complexity
    }
    latency_by_width = {
        float(row["width_mult"]): row
        for row in latency
        if int(row["batch_size"]) == 1
    }
    day33_acc = [float(row["best_val_acc"]) for row in day33]
    day33_time = [float(row["train_seconds"]) for row in day33]

    records = []
    for source in day32:
        width = float(source["width_mult"])
        validate_split(source)
        require_false_test_flag(source)
        if width == 1.0:
            accuracy_mean = sum(day33_acc) / len(day33_acc)
            mean_train_seconds = sum(day33_time) / len(day33_time)
            accuracy_std = statistics.stdev(day33_acc)
            accuracy_source = "Day33 three seeds"
            seed_count = 3
            reliability = "multi_seed"
            seed_text = "42,123,2026"
        else:
            accuracy_mean = float(source["best_val_acc"])
            mean_train_seconds = float(source["train_seconds"])
            accuracy_std = 0.0
            accuracy_source = "Day32 single seed"
            seed_count = 1
            reliability = "single_seed_exploratory"
            seed_text = str(source["training_seed"])

        if width not in complexity_by_width or width not in latency_by_width:
            raise ValueError(f"Missing complexity or latency for width={width}")
        complexity_row = complexity_by_width[width]
        latency_row = latency_by_width[width]
        records.append({
            "model": "Baseline",
            "width_mult": width,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "accuracy_min": accuracy_mean if seed_count == 1 else min(day33_acc),
            "accuracy_max": accuracy_mean if seed_count == 1 else max(day33_acc),
            "accuracy_source": accuracy_source,
            "seed_count": seed_count,
            "training_seeds": seed_text,
            "reliability": reliability,
            "params": int(complexity_row["params"]),
            "params_m": float(complexity_row["params_m"]),
            "macs_m": float(complexity_row["macs_m"]),
            "flops_m": float(complexity_row["flops_m"]),
            "batch1_median_ms": float(latency_row["median_batch_ms"]),
            "batch1_p95_ms": float(latency_row["p95_batch_ms"]),
            "batch1_throughput_images_s": float(
                latency_row["throughput_images_s"]
            ),
            "mean_train_seconds": mean_train_seconds,
            "official_test_evaluated": False,
        })
    return sorted(records, key=lambda row: row["width_mult"])


def is_dominated(candidate, records):
    for other in records:
        if other is candidate:
            continue
        no_worse = (
            other["accuracy_mean"] >= candidate["accuracy_mean"]
            and other["params"] <= candidate["params"]
            and other["flops_m"] <= candidate["flops_m"]
            and other["batch1_median_ms"] <= candidate["batch1_median_ms"]
        )
        strictly_better = (
            other["accuracy_mean"] > candidate["accuracy_mean"]
            or other["params"] < candidate["params"]
            or other["flops_m"] < candidate["flops_m"]
            or other["batch1_median_ms"] < candidate["batch1_median_ms"]
        )
        if no_worse and strictly_better:
            return True
    return False


def build_frontier(records):
    rows = []
    for record in records:
        rows.append({
            "model": record["model"],
            "width_mult": record["width_mult"],
            "accuracy_mean": record["accuracy_mean"],
            "params": record["params"],
            "flops_m": record["flops_m"],
            "batch1_median_ms": record["batch1_median_ms"],
            "reliability": record["reliability"],
            "pareto_optimal": not is_dominated(record, records),
        })
    return rows


def save_pareto_plot(records, path):
    colors = {0.5: "#377eb8", 1.0: "#d95f02", 1.5: "#4daf4a"}
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    for record in records:
        axis.scatter(
            record["params"] / 1000,
            record["accuracy_mean"],
            s=110,
            color=colors[record["width_mult"]],
            edgecolor="black",
            linewidth=0.6,
        )
        label = f"width={record['width_mult']}"
        if record["reliability"] == "single_seed_exploratory":
            label += "*"
        axis.annotate(
            label,
            (record["params"] / 1000, record["accuracy_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.set_xlabel("Trainable parameters (thousands)")
    axis.set_ylabel("Best validation accuracy (%)")
    axis.set_title("Accuracy versus Model Cost")
    axis.grid(alpha=0.25)
    axis.text(
        0.02,
        0.03,
        "* single-seed exploratory result",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    day32 = load_csv(root_dir / args.day32)
    day33 = load_csv(root_dir / args.day33)
    complexity = load_csv(root_dir / args.complexity)
    latency = load_csv(root_dir / args.latency)
    records = make_baseline_records(day32, day33, complexity, latency)
    frontier = build_frontier(records)
    save_csv(records, output_dir / "model_summary.csv")
    save_csv(frontier, output_dir / "pareto_frontier.csv")
    save_pareto_plot(records, output_dir / "accuracy_cost_pareto.png")

    se_stats = load_csv(root_dir / args.se_stats)
    se_row = next(row for row in se_stats if row["model"] == "SE")
    decision = {
        "recommended_model": "Baseline width=1.0",
        "reason": (
            "It is the highest-width candidate with a three-seed result, "
            "while keeping parameters and CPU latency far below width=1.5."
        ),
        "alternative_for_strict_latency": "Baseline width=0.5",
        "not_selected": {
            "width=1.5": "Only one seed and much higher cost; needs a proper multi-seed rerun.",
            "width=1.0+SE": "Mean validation accuracy decreased by 0.35 percentage points.",
        },
        "se_mean_best_val_acc": float(se_row["mean_best_val_acc"]),
        "se_sample_std_best_val_acc": float(se_row["sample_std_best_val_acc"]),
        "official_test_evaluated": False,
        "split_sha256": EXPECTED_SPLIT_SHA256,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Day39: 综合指标汇总与模型选择")
    for record in records:
        print(
            f"width={record['width_mult']} | "
            f"acc={record['accuracy_mean']:.2f}% +/- {record['accuracy_std']:.2f}% | "
            f"params={record['params']:,} | FLOPs={record['flops_m']:.3f}M | "
            f"batch1={record['batch1_median_ms']:.3f}ms | "
            f"reliability={record['reliability']}"
        )
    print("推荐候选: Baseline width=1.0")
    print("official test: NOT EVALUATED")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day32", default="checkpoints/day32_proper_split_models/summary.csv")
    parser.add_argument("--day33", default="checkpoints/day33_multiseed/summary.csv")
    parser.add_argument("--complexity", default="checkpoints/day35_complexity_profile/complexity.csv")
    parser.add_argument("--latency", default="checkpoints/day36_inference_benchmark/benchmark_summary.csv")
    parser.add_argument("--se-stats", default="checkpoints/day38_se_ablation/model_statistics.csv")
    parser.add_argument("--output", default="checkpoints/day39_model_selection")
    args = parser.parse_args()
    # Keep the default path compatible with both the original and current Day32 folder names.
    if not (Path(__file__).resolve().parent.parent / args.day32).exists():
        args.day32 = "checkpoints/day32_proper_width_models/summary.csv"
    main(args)
