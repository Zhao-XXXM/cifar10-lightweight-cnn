import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

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


def parse_widths(text):
    return [float(item.strip()) for item in text.split(",")]


def width_to_name(width):
    return f"width_{str(width).replace('.', '_')}"


def build_transforms():
    display_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    model_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return display_transform, model_transform


def load_sample(root_dir, index, image_path):
    display_transform, model_transform = build_transforms()

    if image_path:
        image = Image.open(image_path).convert("RGB")
        return (
            display_transform(image),
            model_transform(image),
            "unknown",
            f"custom image: {Path(image_path).name}",
        )

    display_dataset = torchvision.datasets.CIFAR10(
        root=str(root_dir / "data"),
        train=False,
        download=False,
        transform=display_transform,
    )
    model_dataset = torchvision.datasets.CIFAR10(
        root=str(root_dir / "data"),
        train=False,
        download=False,
        transform=model_transform,
    )
    if index < 0 or index >= len(display_dataset):
        raise ValueError(f"index 应在 0 到 {len(display_dataset) - 1} 之间")

    display_image, label = display_dataset[index]
    model_image, _ = model_dataset[index]
    return display_image, model_image, CLASSES[label], f"CIFAR-10 test index: {index}"


def predict(model, model_image, device, topk):
    model.eval()
    with torch.no_grad():
        logits = model(model_image.unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=1)[0]
        values, indices = torch.topk(probabilities, k=topk)

    return [
        {
            "rank": rank,
            "class_id": int(class_id),
            "class_name": CLASSES[int(class_id)],
            "probability": float(probability),
        }
        for rank, (probability, class_id) in enumerate(zip(values, indices), start=1)
    ]


def save_results_csv(records, path):
    fields = ["width_mult", "rank", "class_id", "class_name", "probability"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def save_visualization(display_image, results, true_label, sample_title, path):
    figure, axes = plt.subplots(1, len(results) + 1, figsize=(5 + 4 * len(results), 4.5))
    axes = list(axes)

    axes[0].imshow(display_image.permute(1, 2, 0).numpy())
    axes[0].set_title(f"{sample_title}\ntrue: {true_label}")
    axes[0].axis("off")

    for axis, result in zip(axes[1:], results):
        predictions = result["predictions"]
        names = [item["class_name"] for item in predictions][::-1]
        probabilities = [item["probability"] for item in predictions][::-1]
        top1_name = predictions[0]["class_name"]
        colors = ["#d95f02" if name == top1_name else "#377eb8" for name in names]

        axis.barh(names, probabilities, color=colors)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Softmax probability")
        axis.set_title(f"width_mult={result['width_mult']}\nTop-1: {predictions[0]['class_name']}")
        for position, probability in enumerate(probabilities):
            axis.text(min(probability + 0.02, 0.92), position, f"{probability:.3f}", va="center")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="0.5,1.0,1.5")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image", default=None)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if args.topk < 1 or args.topk > 10:
        raise ValueError("topk 必须在 1 到 10 之间")

    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "checkpoints" / "day30_single_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    display_image, model_image, true_label, sample_title = load_sample(
        root_dir, args.index, args.image
    )

    all_records = []
    results = []
    for width in parse_widths(args.widths):
        checkpoint_path = (
            root_dir / "checkpoints" / "day28_width_models" /
            width_to_name(width) / "best.pth"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

        model = WidthLightVGGSlimGAP(width_mult=width, num_classes=10).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        predictions = predict(model, model_image, device, args.topk)
        results.append({"width_mult": width, "predictions": predictions})

        for prediction in predictions:
            all_records.append({"width_mult": width, **prediction})

        top1 = predictions[0]
        print(
            f"width_mult={width} | Top-1={top1['class_name']} "
            f"({top1['probability']:.4f}) | "
            f"Top-{args.topk}={[item['class_name'] for item in predictions]}"
        )

    if args.image:
        output_stem = "custom"
    else:
        output_stem = f"index_{args.index}"
    save_results_csv(all_records, output_dir / f"{output_stem}_predictions.csv")
    save_visualization(
        display_image,
        results,
        true_label,
        sample_title,
        output_dir / f"{output_stem}_predictions.png",
    )
    summary = {
        "sample": sample_title,
        "true_label": true_label,
        "results": results,
    }
    (output_dir / f"{output_stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
