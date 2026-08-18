import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from d20_light_cnn import count_trainable_params
from d32_train_proper_split import (
    build_loaders,
    evaluate,
    load_split,
    set_seed,
    train_one_epoch,
)
from d37_se_attention import SEWidthLightVGGSlimGAP


def parse_seeds(text):
    seeds = [int(item.strip()) for item in text.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds 必须是非空且不重复的整数列表")
    return seeds


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_se(seed, args, root_dir, data_dir, split_data, output_root, device):
    set_seed(seed)
    train_loader, val_loader = build_loaders(
        data_dir,
        split_data,
        args.batch_size,
        args.num_workers,
        seed,
        device,
    )
    model = SEWidthLightVGGSlimGAP(
        width_mult=1.0,
        num_classes=10,
        reduction=args.reduction,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pth"
    history_path = output_dir / "history.json"

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    start_seconds = time.perf_counter()

    print(f"\n[SE seed={seed}] params={count_trainable_params(model):,}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    train_seconds = time.perf_counter() - start_seconds
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    params = count_trainable_params(model)
    return {
        "model": "SE",
        "training_seed": seed,
        "width_mult": 1.0,
        "reduction": args.reduction,
        "params": params,
        "split_seed": split_data["seed"],
        "split_sha256": split_data["sha256"],
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_acc": history[-1]["val_acc"],
        "train_seconds": train_seconds,
        "official_test_evaluated": False,
        "history_path": str(history_path.relative_to(root_dir)),
        "checkpoint_path": str(checkpoint_path.relative_to(root_dir)),
    }


def load_baseline(path, args, split_data):
    if not path.exists():
        raise FileNotFoundError(f"找不到 Day33 Baseline 汇总: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    if len(records) != 3:
        raise ValueError("Day33 Baseline 必须包含三个训练种子")

    for record in records:
        if (
            float(record["width_mult"]) != 1.0
            or int(record["epochs"]) != args.epochs
            or int(record["batch_size"]) != args.batch_size
            or float(record["lr"]) != args.lr
            or record["split_sha256"] != split_data["sha256"]
            or record["official_test_evaluated"].lower() != "false"
        ):
            raise ValueError("Day33 Baseline 与 SE 实验配置不一致")
        record["model"] = "Baseline"
        record["reduction"] = ""
        record["training_seed"] = int(record["training_seed"])
        record["best_val_acc"] = float(record["best_val_acc"])
        record["final_val_acc"] = float(record["final_val_acc"])
        record["train_seconds"] = float(record["train_seconds"])
        record["params"] = int(record["params"])
    return records


def build_comparison(baseline, se_records, output_dir):
    baseline_by_seed = {record["training_seed"]: record for record in baseline}
    se_by_seed = {record["training_seed"]: record for record in se_records}
    if set(baseline_by_seed) != set(se_by_seed):
        raise ValueError("Baseline 与 SE 的训练种子不一致")

    paired_rows = []
    for seed in sorted(baseline_by_seed):
        base = baseline_by_seed[seed]
        se = se_by_seed[seed]
        paired_rows.append({
            "training_seed": seed,
            "baseline_best_val_acc": base["best_val_acc"],
            "se_best_val_acc": se["best_val_acc"],
            "delta_best_val_acc": se["best_val_acc"] - base["best_val_acc"],
            "baseline_params": base["params"],
            "se_params": se["params"],
            "baseline_train_seconds": base["train_seconds"],
            "se_train_seconds": se["train_seconds"],
        })
    save_csv(paired_rows, output_dir / "paired_comparison.csv")

    summary_rows = []
    for name, records in (("Baseline", baseline), ("SE", se_records)):
        values = [record["best_val_acc"] for record in records]
        summary_rows.append({
            "model": name,
            "n": len(values),
            "mean_best_val_acc": statistics.mean(values),
            "sample_std_best_val_acc": statistics.stdev(values),
            "min_best_val_acc": min(values),
            "max_best_val_acc": max(values),
            "mean_train_seconds": statistics.mean(
                [record["train_seconds"] for record in records]
            ),
        })
    save_csv(summary_rows, output_dir / "model_statistics.csv")

    baseline_mean = summary_rows[0]["mean_best_val_acc"]
    se_mean = summary_rows[1]["mean_best_val_acc"]
    figure, axis = plt.subplots(figsize=(6, 4.5))
    labels = ["Baseline", "SE"]
    means = [row["mean_best_val_acc"] for row in summary_rows]
    errors = [row["sample_std_best_val_acc"] for row in summary_rows]
    axis.bar(labels, means, yerr=errors, capsize=8, color=["#377eb8", "#d95f02"])
    axis.set_ylabel("Best validation accuracy (%)")
    axis.set_title("Baseline vs SE (mean +/- sample std)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "baseline_se_comparison.png", dpi=160)
    plt.close(figure)
    return se_mean - baseline_mean


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    split_path = root_dir / args.split_file
    split_data = load_split(split_path, dataset_size=50000)
    data_dir = root_dir / args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = parse_seeds(args.seeds)
    output_root = root_dir / args.run_name
    se_dir = output_root / "se"
    output_root.mkdir(parents=True, exist_ok=True)

    train_args = SimpleNamespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        reduction=args.reduction,
    )
    se_records = []
    for seed in seeds:
        se_records.append(
            train_se(seed, train_args, root_dir, data_dir, split_data, se_dir, device)
        )
        save_csv(se_records, output_root / "se_summary.csv")

    if not args.skip_comparison:
        baseline_path = root_dir / args.baseline_summary
        baseline_records = load_baseline(baseline_path, train_args, split_data)
        delta = build_comparison(baseline_records, se_records, output_root)
        print(f"Mean SE - Baseline best Val Acc delta: {delta:+.2f} percentage points")

    print("Day38: SE 消融训练")
    print(f"device={device}, seeds={seeds}, split_sha256={split_data['sha256']}")
    print("official test: NOT EVALUATED")
    print(f"结果目录: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="CIFAR-10 数据目录，目录下应包含 cifar-10-batches-py",
    )
    parser.add_argument("--reduction", type=int, default=16)
    parser.add_argument(
        "--split-file",
        default="checkpoints/day31_data_split/seed_42/split_indices.json",
    )
    parser.add_argument(
        "--baseline-summary",
        default="checkpoints/day33_multiseed/summary.csv",
    )
    parser.add_argument("--run-name", default="checkpoints/day38_se_ablation")
    parser.add_argument("--skip-comparison", action="store_true")
    main(parser.parse_args())
