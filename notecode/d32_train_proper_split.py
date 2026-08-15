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
from d31_data_split import split_hash, validate_split


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_widths(text):
    widths = [float(item.strip()) for item in text.split(",")]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("width_mult 必须大于 0")
    return widths


def width_to_name(width):
    return f"width_{str(width).replace('.', '_')}"


def load_split(path, dataset_size):
    with path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    validate_split(train_indices, val_indices, dataset_size)
    actual_hash = split_hash(train_indices, val_indices)
    if actual_hash != split_data["sha256"]:
        raise ValueError("数据划分 SHA-256 与索引内容不一致")
    return split_data


def build_loaders(data_dir, split_data, batch_size, num_workers, loader_seed, device):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_source = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=train_transform
    )
    val_source = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=eval_transform
    )
    train_dataset = torch.utils.data.Subset(
        train_source, split_data["train_indices"]
    )
    val_dataset = torch.utils.data.Subset(
        val_source, split_data["val_indices"]
    )

    generator = torch.Generator().manual_seed(loader_seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


def save_summary(records, path):
    fields = [
        "width_mult",
        "channels",
        "split_seed",
        "split_sha256",
        "training_seed",
        "train_size",
        "val_size",
        "epochs",
        "batch_size",
        "lr",
        "params",
        "checkpoint_mb",
        "best_val_acc",
        "best_epoch",
        "final_val_acc",
        "train_seconds",
        "official_test_evaluated",
        "history_path",
        "checkpoint_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_width(width, args, root_dir, split_data, output_root, device):
    set_seed(args.seed)
    train_loader, val_loader = build_loaders(
        root_dir / "data",
        split_data,
        args.batch_size,
        args.num_workers,
        args.seed,
        device,
    )
    model = WidthLightVGGSlimGAP(width_mult=width, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    output_dir = output_root / width_to_name(width)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pth"
    history_path = output_dir / "history.json"
    params = count_trainable_params(model)
    history = []
    best_val_acc = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    print(f"\n[width_mult={width}] channels={model.channels}, params={params:,}")
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

    train_seconds = time.perf_counter() - start_time
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "width_mult": width,
        "channels": str(model.channels),
        "split_seed": split_data["seed"],
        "split_sha256": split_data["sha256"],
        "training_seed": args.seed,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "params": params,
        "checkpoint_mb": checkpoint_path.stat().st_size / 1024 / 1024,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_acc": history[-1]["val_acc"],
        "train_seconds": train_seconds,
        "official_test_evaluated": False,
        "history_path": str(history_path.relative_to(root_dir)),
        "checkpoint_path": str(checkpoint_path.relative_to(root_dir)),
    }


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    split_path = root_dir / args.split_file
    output_root = root_dir / "checkpoints" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    split_data = load_split(split_path, dataset_size=50000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    widths = parse_widths(args.widths)
    print("Day32: 规范 Train/Validation 宽度对照实验")
    print(
        f"device={device}, train={split_data['train_size']}, "
        f"val={split_data['val_size']}"
    )
    print(f"split_sha256={split_data['sha256']}")
    print("official test: NOT EVALUATED")

    records = []
    summary_path = output_root / "summary.csv"
    for width in widths:
        records.append(
            run_width(width, args, root_dir, split_data, output_root, device)
        )
        save_summary(records, summary_path)

    print("\n[Day32 Validation 汇总]")
    for record in records:
        print(
            f"width={record['width_mult']} | params={record['params']:,} | "
            f"best_val_acc={record['best_val_acc']:.2f}% | "
            f"best_epoch={record['best_epoch']}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--widths", default="0.5,1.0,1.5")
    parser.add_argument(
        "--split-file",
        default="checkpoints/day31_data_split/seed_42/split_indices.json",
    )
    parser.add_argument("--run-name", default="day32_proper_width_models")
    main(parser.parse_args())
