import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torchvision


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def stratified_split(targets, val_per_class, seed):
    class_indices = {class_id: [] for class_id in range(len(CLASSES))}
    for index, class_id in enumerate(targets):
        class_indices[class_id].append(index)

    generator = random.Random(seed)
    train_indices = []
    val_indices = []
    for class_id in range(len(CLASSES)):
        indices = class_indices[class_id]
        if val_per_class <= 0 or val_per_class >= len(indices):
            raise ValueError(
                f"val_per_class 必须在 1 到 {len(indices) - 1} 之间"
            )
        generator.shuffle(indices)
        val_indices.extend(indices[:val_per_class])
        train_indices.extend(indices[val_per_class:])

    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    return train_indices, val_indices


def validate_split(train_indices, val_indices, dataset_size):
    train_set = set(train_indices)
    val_set = set(val_indices)
    if len(train_set) != len(train_indices):
        raise ValueError("训练集索引存在重复")
    if len(val_set) != len(val_indices):
        raise ValueError("验证集索引存在重复")
    if not train_set.isdisjoint(val_set):
        raise ValueError("训练集和验证集存在重叠")
    if train_set | val_set != set(range(dataset_size)):
        raise ValueError("训练集和验证集没有完整覆盖官方训练集")


def count_by_class(targets, indices=None):
    if indices is None:
        selected_targets = targets
    else:
        selected_targets = [targets[index] for index in indices]
    return Counter(selected_targets)


def split_hash(train_indices, val_indices):
    compact_json = json.dumps(
        {"train_indices": train_indices, "val_indices": val_indices},
        separators=(",", ":"),
    )
    return hashlib.sha256(compact_json.encode("utf-8")).hexdigest()


def save_class_distribution(train_counts, val_counts, test_counts, path):
    rows = []
    for class_id, class_name in enumerate(CLASSES):
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "train_count": train_counts[class_id],
            "val_count": val_counts[class_id],
            "test_count": test_counts[class_id],
        })

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    output_dir = root_dir / "checkpoints" / "day31_data_split" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    official_train = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    official_test = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    train_indices, val_indices = stratified_split(
        official_train.targets,
        args.val_per_class,
        args.seed,
    )
    validate_split(train_indices, val_indices, len(official_train))

    train_counts = count_by_class(official_train.targets, train_indices)
    val_counts = count_by_class(official_train.targets, val_indices)
    test_counts = count_by_class(official_test.targets)
    digest = split_hash(train_indices, val_indices)

    split_data = {
        "seed": args.seed,
        "val_per_class": args.val_per_class,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "official_test_size": len(official_test),
        "sha256": digest,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    (output_dir / "split_indices.json").write_text(
        json.dumps(split_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_class_distribution(
        train_counts,
        val_counts,
        test_counts,
        output_dir / "class_distribution.csv",
    )

    print("Day31: CIFAR-10 分层数据划分")
    print(f"seed={args.seed}, sha256={digest}")
    print(
        f"train={len(train_indices)}, val={len(val_indices)}, "
        f"official_test={len(official_test)}"
    )
    for class_id, class_name in enumerate(CLASSES):
        print(
            f"{class_name:10s} | train={train_counts[class_id]:4d} | "
            f"val={val_counts[class_id]:3d} | test={test_counts[class_id]:4d}"
        )
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=500)
    main(parser.parse_args())
