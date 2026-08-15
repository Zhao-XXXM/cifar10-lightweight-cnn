import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from d32_train_proper_split import load_split, run_width, save_summary


def parse_seeds(text):
    seeds = [int(item.strip()) for item in text.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds 必须是非空且不重复的整数列表")
    return seeds


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    split_path = root_dir / args.split_file
    split_data = load_split(split_path, dataset_size=50000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = parse_seeds(args.seeds)
    output_root = root_dir / "checkpoints" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("Day33: 多随机种子 Validation 稳定性实验")
    print(f"device={device}, width_mult={args.width}")
    print(f"seeds={seeds}, split_sha256={split_data['sha256']}")
    print("official test: NOT EVALUATED")

    all_records = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_args = SimpleNamespace(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
        )
        record = run_width(
            args.width,
            seed_args,
            root_dir,
            split_data,
            seed_dir,
            device,
        )
        all_records.append(record)
        save_summary([record], seed_dir / "summary.csv")
        save_summary(all_records, output_root / "summary.csv")

    print("\n[Day33 原始结果]")
    for record in all_records:
        print(
            f"training_seed={record['training_seed']} | "
            f"best_val_acc={record['best_val_acc']:.2f}% | "
            f"best_epoch={record['best_epoch']} | "
            f"time={record['train_seconds']:.2f}s"
        )
    print(f"summary: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--split-file",
        default="checkpoints/day31_data_split/seed_42/split_indices.json",
    )
    parser.add_argument("--run-name", default="day33_multiseed")
    main(parser.parse_args())
