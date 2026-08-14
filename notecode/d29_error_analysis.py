import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

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
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
STD = torch.tensor((0.2023, 0.1994, 0.2010)).view(3, 1, 1)


def parse_widths(text):
    return [float(item.strip()) for item in text.split(",")]


def width_to_name(width):
    return f"width_{str(width).replace('.', '_')}"


def build_test_loader(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def collect_predictions(model, loader, device, max_errors):
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    errors = []
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            predictions = logits.argmax(dim=1)

            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                confusion[true_label, predicted_label] += 1

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for index in range(labels.size(0)):
                if predictions[index] != labels[index]:
                    errors.append((
                        probabilities[index, predictions[index]].item(),
                        images[index].cpu(),
                        labels[index].item(),
                        predictions[index].item(),
                    ))

    errors.sort(key=lambda item: item[0], reverse=True)
    return confusion, errors[:max_errors], correct, total


def save_class_metrics(confusion, output_path):
    rows = []
    for index, class_name in enumerate(CLASSES):
        total = int(confusion[index].sum().item())
        correct = int(confusion[index, index].item())
        accuracy = 100.0 * correct / total if total else 0.0
        rows.append({
            "class_id": index,
            "class_name": class_name,
            "samples": total,
            "correct": correct,
            "accuracy": accuracy,
        })

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_confused_pairs(confusion, output_path):
    pairs = []
    for true_index in range(10):
        for predicted_index in range(10):
            if true_index != predicted_index:
                pairs.append({
                    "true_class": CLASSES[true_index],
                    "predicted_class": CLASSES[predicted_index],
                    "count": int(confusion[true_index, predicted_index].item()),
                })

    pairs.sort(key=lambda item: item["count"], reverse=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=pairs[0].keys())
        writer.writeheader()
        writer.writerows(pairs)
    return pairs


def save_confusion_matrix(confusion, output_path, width):
    matrix = confusion.numpy()
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(10), CLASSES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"CIFAR-10 Confusion Matrix (width_mult={width})")

    for row in range(10):
        for column in range(10):
            color = "white" if normalized[row, column] > 0.5 else "black"
            axis.text(column, row, f"{normalized[row, column]:.2f}",
                      ha="center", va="center", color=color, fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def unnormalize(image):
    return (image * STD + MEAN).clamp(0.0, 1.0)


def save_error_grid(errors, output_path, width):
    if not errors:
        return

    columns = 4
    rows = (len(errors) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10, 2.8 * rows))
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(errors):
            continue
        confidence, image, true_label, predicted_label = errors[index]
        axis.imshow(unnormalize(image).permute(1, 2, 0).numpy())
        axis.set_title(
            f"true: {CLASSES[true_label]}\n"
            f"pred: {CLASSES[predicted_label]} ({confidence:.2f})",
            fontsize=9,
        )

    figure.suptitle(f"High-confidence errors (width_mult={width})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def analyze_width(width, args, root_dir, loader, device):
    checkpoint_dir = root_dir / "checkpoints" / "day28_width_models" / width_to_name(width)
    checkpoint_path = checkpoint_dir / "best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    model = WidthLightVGGSlimGAP(width_mult=width, num_classes=10).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    confusion, errors, correct, total = collect_predictions(
        model, loader, device, args.max_errors
    )
    output_dir = root_dir / "checkpoints" / "day29_error_analysis" / width_to_name(width)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_rows = save_class_metrics(confusion, output_dir / "class_metrics.csv")
    pair_rows = save_confused_pairs(confusion, output_dir / "confused_pairs.csv")
    save_confusion_matrix(confusion, output_dir / "confusion_matrix.png", width)
    save_error_grid(errors, output_dir / "error_examples.png", width)

    worst_class = min(class_rows, key=lambda row: row["accuracy"])
    summary = {
        "width_mult": width,
        "correct": correct,
        "total": total,
        "accuracy": 100.0 * correct / total,
        "worst_class": worst_class["class_name"],
        "worst_class_accuracy": worst_class["accuracy"],
        "top_confused_pair": pair_rows[0],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"width_mult={width} | accuracy={summary['accuracy']:.2f}% | "
        f"worst_class={summary['worst_class']} "
        f"({summary['worst_class_accuracy']:.2f}%) | "
        f"top_pair={pair_rows[0]['true_class']} -> "
        f"{pair_rows[0]['predicted_class']} ({pair_rows[0]['count']})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="1.0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=12)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_test_loader(root_dir / "data", args.batch_size, args.num_workers)
    print(f"device={device}")

    for width in parse_widths(args.widths):
        analyze_width(width, args, root_dir, loader, device)


if __name__ == "__main__":
    main()
