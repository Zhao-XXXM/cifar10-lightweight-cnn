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

from d20_light_cnn import LightVGGSlim, count_trainable_params


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


EXPERIMENTS = {
    "baseline": {
        "use_augmentation": False,
        "weight_decay": 0.0,
        "use_cosine": False,
    },
    "augmentation": {
        "use_augmentation": True,
        "weight_decay": 0.0,
        "use_cosine": False,
    },
    "weight_decay": {
        "use_augmentation": True,
        "weight_decay": 5e-4,
        "use_cosine": False,
    },
    "cosine": {
        "use_augmentation": True,
        "weight_decay": 5e-4,
        "use_cosine": True,
    },
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, batch_size, use_augmentation):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
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


def run_experiment(name, config, args, root_dir, device):
    experiment_dir = root_dir / "checkpoints" / "day23_ablation" / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    train_loader, test_loader = build_loaders(
        root_dir / "data",
        args.batch_size,
        config["use_augmentation"],
    )

    model = LightVGGSlim(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config["weight_decay"],
    )
    scheduler = None
    if config["use_cosine"]:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    history = []
    best_acc = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    print(f"\n[{name}]")
    print(
        f"augmentation={config['use_augmentation']}, "
        f"weight_decay={config['weight_decay']}, "
        f"cosine={config['use_cosine']}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), experiment_dir / "best.pth")

        if scheduler is not None:
            scheduler.step()

    elapsed = time.perf_counter() - start_time
    history_path = experiment_dir / "history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "experiment": name,
        "augmentation": config["use_augmentation"],
        "weight_decay": config["weight_decay"],
        "cosine_scheduler": config["use_cosine"],
        "params": count_trainable_params(model),
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "train_seconds": elapsed,
        "history_path": str(history_path.relative_to(root_dir)),
    }


def save_summary(results, path):
    fields = [
        "experiment",
        "augmentation",
        "weight_decay",
        "cosine_scheduler",
        "params",
        "best_val_acc",
        "best_epoch",
        "train_seconds",
        "history_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main(args):
    root_dir = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiments == "all":
        names = list(EXPERIMENTS)
    else:
        names = [name.strip() for name in args.experiments.split(",")]
        unknown = [name for name in names if name not in EXPERIMENTS]
        if unknown:
            raise ValueError(
                f"未知实验: {unknown}，可选值为 {list(EXPERIMENTS)}"
            )

    print("Day23: LightVGG-Slim 消融实验")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"实验列表: {names}")

    results = []
    for name in names:
        result = run_experiment(
            name,
            EXPERIMENTS[name],
            args,
            root_dir,
            device,
        )
        results.append(result)

    output_dir = root_dir / "checkpoints" / "day23_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    save_summary(results, summary_path)

    print("\n[消融实验汇总]")
    for result in results:
        print(
            f"{result['experiment']:12s} | "
            f"Best Val Acc: {result['best_val_acc']:.2f}% | "
            f"Best Epoch: {result['best_epoch']} | "
            f"Time: {result['train_seconds']:.2f}s"
        )
    print(f"\n汇总表保存至: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        default="all",
        help="all 或逗号分隔的实验名",
    )
    main(parser.parse_args())
