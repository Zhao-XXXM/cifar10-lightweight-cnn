import csv
import json
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from d10_vgg_slim import VGGSlim
from d20_light_cnn import LightVGGSlim, count_trainable_params
from d24_light_gap import LightVGGSlimGAP


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def checkpoint_size_mb(path):
    if not path.exists():
        return None
    return path.stat().st_size / 1024 / 1024


def load_history(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    if not history:
        raise ValueError(f"历史记录为空: {path}")
    return history


def get_best_result(history):
    return max(history, key=lambda item: item["val_acc"])


def build_model(name):
    models = {
        "VGG-Slim": VGGSlim(num_classes=10),
        "LightVGG-Slim": LightVGGSlim(num_classes=10),
        "LightVGG-Slim-GAP": LightVGGSlimGAP(num_classes=10),
    }
    return models[name]


def build_test_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def evaluate_checkpoint(model, checkpoint_path, test_loader, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return 100.0 * correct / total


def build_records(root_dir):
    checkpoint_dir = root_dir / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = build_test_loader(root_dir / "data", batch_size=128)
    configs = [
        {
            "model": "VGG-Slim",
            "history": checkpoint_dir / "vgg_slim_history.json",
            "checkpoint": checkpoint_dir / "vgg_slim_best.pth",
        },
        {
            "model": "LightVGG-Slim",
            "history": checkpoint_dir / "light_vgg_slim_history.json",
            "checkpoint": checkpoint_dir / "light_vgg_slim_best.pth",
        },
        {
            "model": "LightVGG-Slim-GAP",
            "history": checkpoint_dir / "light_vgg_gap_history.json",
            "checkpoint": checkpoint_dir / "light_vgg_gap_best.pth",
        },
    ]

    records = []
    for config in configs:
        model = build_model(config["model"])
        history = load_history(config["history"])
        params = count_trainable_params(model)
        params_m = params / 1e6
        checkpoint_mb = checkpoint_size_mb(config["checkpoint"])

        if history is None:
            best_val_acc = evaluate_checkpoint(
                model,
                config["checkpoint"],
                test_loader,
                device,
            )
            best_epoch = "not_recorded"
            final_val_acc = best_val_acc
            epochs = "not_recorded"
            result_source = "checkpoint_eval"
        else:
            best = get_best_result(history)
            best_val_acc = best["val_acc"]
            best_epoch = best["epoch"]
            final_val_acc = history[-1]["val_acc"]
            epochs = len(history)
            result_source = "history_json"

        records.append({
            "model": config["model"],
            "params": params,
            "params_m": params_m,
            "checkpoint_mb": checkpoint_mb,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "final_val_acc": final_val_acc,
            "epochs": epochs,
            "train_seconds": "not_recorded",
            "result_source": result_source,
            "acc_per_million_params": best_val_acc / params_m,
        })
    return records


def save_csv(records, path):
    fields = [
        "model",
        "params",
        "params_m",
        "checkpoint_mb",
        "best_val_acc",
        "best_epoch",
        "final_val_acc",
        "epochs",
        "train_seconds",
        "result_source",
        "acc_per_million_params",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def save_markdown(records, path):
    baseline = records[0]
    lines = [
        "# 三模型对照实验",
        "",
        "| 模型 | 参数量 | 参数量(M) | Checkpoint(MB) | Val Acc | 最佳 Epoch | 最终 Val Acc | 数据来源 | Acc/百万参数 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]

    for record in records:
        lines.append(
            f"| {record['model']} | "
            f"{record['params']:,} | "
            f"{record['params_m']:.4f} | "
            f"{record['checkpoint_mb']:.2f} | "
            f"{record['best_val_acc']:.2f}% | "
            f"{record['best_epoch']} | "
            f"{record['final_val_acc']:.2f}% | "
            f"{record['result_source']} | "
            f"{record['acc_per_million_params']:.2f} |"
        )

    lines.extend([
        "",
        "## 相对 VGG-Slim 的变化",
        "",
        "| 模型 | 参数减少比例 | 最佳准确率差值 |",
        "|---|---:|---:|",
    ])

    for record in records:
        reduction = 100.0 * (1.0 - record["params"] / baseline["params"])
        acc_delta = record["best_val_acc"] - baseline["best_val_acc"]
        lines.append(
            f"| {record['model']} | {reduction:.2f}% | {acc_delta:+.2f} 个百分点 |"
        )

    lines.extend([
        "",
        "> VGG-Slim 缺少 history JSON，因此表中 VGG-Slim 的 Val Acc 来自已保存 checkpoint 的重新评估；最佳 Epoch 与训练时间不补估计值。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "checkpoints" / "day26_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(root)
    csv_path = output_dir / "model_comparison.csv"
    markdown_path = output_dir / "model_comparison.md"
    save_csv(records, csv_path)
    save_markdown(records, markdown_path)

    print("Day26: 三模型对照实验")
    print("=" * 60)
    for record in records:
        print(
            f"{record['model']:18s} | "
            f"Params: {record['params']:,} | "
            f"Best Val Acc: {record['best_val_acc']:.2f}% | "
            f"Checkpoint: {record['checkpoint_mb']:.2f} MB"
        )
    print(f"\nCSV 保存至: {csv_path}")
    print(f"Markdown 保存至: {markdown_path}")
