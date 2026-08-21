import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from d27_width_multiplier import WidthLightVGGSlimGAP


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
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
SEEDS = (42, 123, 2026)


def width_to_name(width):
    return f"width_{str(width).replace('.', '_')}"


def build_loader(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_sum += loss_fn(logits, labels).item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                confusion[true_label, predicted_label] += 1

    class_totals = confusion.sum(dim=1)
    class_correct = confusion.diag()
    class_accuracy = [
        100.0 * class_correct[index].item() / class_totals[index].item()
        for index in range(10)
    ]
    return {
        "test_loss": loss_sum / total,
        "test_accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "class_accuracy": class_accuracy,
        "confusion": confusion,
    }


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_csv(confusion, path):
    rows = []
    for true_index, class_name in enumerate(CLASSES):
        row = {"true_class": class_name}
        row.update({
            f"pred_{predicted_name}": int(confusion[true_index, predicted_index])
            for predicted_index, predicted_name in enumerate(CLASSES)
        })
        rows.append(row)
    save_csv(rows, path)


def save_confusion_plot(confusion, path):
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(10), CLASSES, rotation=35, ha="right")
    axis.set_yticks(range(10), CLASSES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Aggregated Confusion Matrix: Final width=1.0 Model")
    threshold = confusion.max().item() * 0.55
    for row in range(10):
        for column in range(10):
            value = int(confusion[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / args.data_dir
    output_dir = root_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(data_dir, args.batch_size, args.num_workers)

    records = []
    total_confusion = torch.zeros(10, 10, dtype=torch.int64)
    for seed in SEEDS:
        checkpoint = (
            root_dir
            / args.checkpoint_root
            / f"seed_{seed}"
            / width_to_name(args.width)
            / "best.pth"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        model = WidthLightVGGSlimGAP(
            width_mult=args.width,
            num_classes=10,
        ).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        result = evaluate(model, loader, device)
        total_confusion += result["confusion"]
        records.append({
            "model": "Baseline",
            "width_mult": args.width,
            "training_seed": seed,
            "checkpoint": str(checkpoint.relative_to(root_dir)),
            "test_loss": result["test_loss"],
            "test_accuracy": result["test_accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "official_test_evaluated": True,
        })
        print(
            f"seed={seed} | Test Loss={result['test_loss']:.4f} | "
            f"Test Acc={result['test_accuracy']:.2f}%"
        )

    values = [row["test_accuracy"] for row in records]
    summary = {
        "model": "Baseline",
        "width_mult": args.width,
        "training_seeds": list(SEEDS),
        "test_accuracy_mean": statistics.mean(values),
        "test_accuracy_sample_std": statistics.stdev(values),
        "test_accuracy_min": min(values),
        "test_accuracy_max": max(values),
        "test_size": records[0]["total"],
        "official_test_evaluated": True,
        "selection_was_completed_before_test": True,
        "device": str(device),
    }
    class_rows = []
    totals = total_confusion.sum(dim=1)
    correct = total_confusion.diag()
    for index, class_name in enumerate(CLASSES):
        class_rows.append({
            "class_id": index,
            "class_name": class_name,
            "correct": int(correct[index]),
            "total": int(totals[index]),
            "accuracy": 100.0 * correct[index].item() / totals[index].item(),
        })

    save_csv(records, output_dir / "test_summary.csv")
    save_csv(class_rows, output_dir / "class_accuracy.csv")
    save_confusion_csv(total_confusion, output_dir / "confusion_matrix.csv")
    save_confusion_plot(
        total_confusion,
        output_dir / "confusion_matrix.png",
    )
    (output_dir / "final_report.json").write_text(
        json.dumps({"summary": summary, "class_accuracy": class_rows}, indent=2),
        encoding="utf-8",
    )

    print(
        f"Final Test Acc = {summary['test_accuracy_mean']:.2f}% +/- "
        f"{summary['test_accuracy_sample_std']:.2f}%"
    )
    print("官方 Test Set：已进行最终评估，未用于模型选择")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/day33_multiseed",
    )
    parser.add_argument("--output", default="checkpoints/day40_final_evaluation")
    main(parser.parse_args())
