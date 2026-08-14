import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from d20_light_cnn import count_trainable_params
from d27_width_multiplier import WidthLightVGGSlimGAP


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_widths(widths_text):
    widths = []
    for item in widths_text.split(","):
        width = float(item.strip())
        if width <= 0:
            raise ValueError("width_mult 必须大于 0")
        widths.append(width)
    return widths


def width_to_name(width):
    return f"width_{str(width).replace('.', '_')}"


def build_loaders(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    return loss_sum / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return loss_sum / total, 100.0 * correct / total


def checkpoint_size_mb(path):
    if not path.exists():
        return 0.0
    return path.stat().st_size / 1024 / 1024


def save_summary(results, path):
    fields = [
        "width_mult",
        "channels",
        "epochs",
        "batch_size",
        "lr",
        "seed",
        "params",
        "params_m",
        "checkpoint_mb",
        "best_val_acc",
        "best_epoch",
        "final_val_acc",
        "train_seconds",
        "acc_per_million_params",
        "history_path",
        "checkpoint_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def run_width_experiment(width, args, root_dir, device):
    set_seed(args.seed)
    train_loader, test_loader = build_loaders(
        root_dir / "data",
        args.batch_size,
        args.num_workers,
    )

    model = WidthLightVGGSlimGAP(width_mult=width, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    output_dir = root_dir / "checkpoints" / "day28_width_models" / width_to_name(width)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pth"
    history_path = output_dir / "history.json"

    params = count_trainable_params(model)
    history = []
    best_acc = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    print(f"\n[width_mult={width}]")
    print(f"channels={model.channels}, params={params:,}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

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

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    train_seconds = time.perf_counter() - start_time
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_acc = history[-1]["val_acc"]
    params_m = params / 1e6
    return {
        "width_mult": width,
        "channels": str(model.channels),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "params": params,
        "params_m": params_m,
        "checkpoint_mb": checkpoint_size_mb(checkpoint_path),
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "final_val_acc": final_acc,
        "train_seconds": train_seconds,
        "acc_per_million_params": best_acc / params_m,
        "history_path": str(history_path.relative_to(root_dir)),
        "checkpoint_path": str(checkpoint_path.relative_to(root_dir)),
    }


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "checkpoints" / "day28_width_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    widths = parse_widths(args.widths)

    print("Day28: Width Multiplier 训练对照实验")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"lr={args.lr}, seed={args.seed}, widths={widths}")

    results = []
    for width in widths:
        result = run_width_experiment(width, args, root_dir, device)
        results.append(result)

    summary_path = output_dir / "summary.csv"
    save_summary(results, summary_path)

    print("\n[Day28 汇总]")
    for result in results:
        print(
            f"width={result['width_mult']} | "
            f"Params: {result['params']:,} | "
            f"Best Val Acc: {result['best_val_acc']:.2f}% | "
            f"Checkpoint: {result['checkpoint_mb']:.2f} MB | "
            f"Time: {result['train_seconds']:.2f}s"
        )
    print(f"\nsummary 保存至: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--widths",
        default="0.5,1.0,1.5",
        help="逗号分隔的 width multiplier，例如 0.5,1.0,1.5",
    )
    main(parser.parse_args())
