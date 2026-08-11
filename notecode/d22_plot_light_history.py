import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(path):
    with path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    if not history:
        raise ValueError("history 文件为空，无法绘图")
    return history


def save_csv(history, path):
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path):
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_acc = [item["val_acc"] for item in history]

    best_item = max(history, key=lambda item: item["val_acc"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="s", label="Val Loss")
    axes[0].set_title("LightVGG-Slim Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Train Acc")
    axes[1].plot(epochs, val_acc, marker="^", label="Val Acc")
    axes[1].axhline(
        best_item["val_acc"],
        color="green",
        linestyle="--",
        alpha=0.6,
        label=f"Best Val Acc: {best_item['val_acc']:.2f}%",
    )
    axes[1].set_title("LightVGG-Slim Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        default="checkpoints/light_vgg_slim_history.json",
    )
    parser.add_argument(
        "--plot",
        default="light_vgg_slim_training_curves.png",
    )
    parser.add_argument(
        "--csv",
        default="checkpoints/light_vgg_slim_history.csv",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    history_path = root_dir / args.history
    plot_path = root_dir / args.plot
    csv_path = root_dir / args.csv

    history = load_history(history_path)
    save_csv(history, csv_path)
    plot_history(history, plot_path)

    best_item = max(history, key=lambda item: item["val_acc"])
    final_item = history[-1]
    generalization_gap = final_item["train_acc"] - final_item["val_acc"]

    print("Day22: LightVGG-Slim 实验记录整理")
    print(f"训练 Epoch 数: {len(history)}")
    print(
        f"最佳 Val Acc: {best_item['val_acc']:.2f}% "
        f"(Epoch {best_item['epoch']})"
    )
    print(f"最终 Train Acc: {final_item['train_acc']:.2f}%")
    print(f"最终 Val Acc: {final_item['val_acc']:.2f}%")
    print(f"最终 Train-Val Acc 差值: {generalization_gap:.2f}%")
    print(f"曲线保存至: {plot_path}")
    print(f"CSV 保存至: {csv_path}")


if __name__ == "__main__":
    main()
