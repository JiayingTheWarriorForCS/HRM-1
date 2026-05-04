import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader

CHECKPOINT_DIR = "/content/drive/MyDrive/HRM_outputs/less_size_run"
CHECKPOINT_FILE = "step_14960"

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # ===== load config =====
    with open(os.path.join(CHECKPOINT_DIR, "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    config.checkpoint_path = CHECKPOINT_DIR

    # ===== dataloader =====
    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0, world_size=1
    )

    eval_loader, eval_metadata = create_dataloader(
        config, "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0, world_size=1
    )

    # ===== load model =====
    state = init_train_state(config, train_metadata, world_size=1)
    state.model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE), map_location=device)
    )

    model = state.model.to(device)
    model.eval()

    # ===== HRM evaluation =====
    print("Running HRM evaluation...")
    metrics = evaluate(config, state, eval_loader, eval_metadata, rank=0, world_size=1)

    print("\n===== HRM Metrics =====")
    print(metrics)

    # ===== 简单 baseline（V-JEPA）=====
    print("\nRunning baseline...")

    mse_list = []

    for _, batch, _ in eval_loader:
        data = batch

        inputs = data["inputs"].to(device)
        labels = data["labels"].to(device)

        # baseline: 不做推理（identity）
        pred = inputs

        mse = torch.mean((pred - labels) ** 2).item()
        mse_list.append(mse)

    mse_vjepa = np.mean(mse_list)

    print("\n===== Final Comparison =====")
    print(f"HRM MSE: {metrics['all']['mse']:.4f}")
    print(f"Baseline MSE: {mse_vjepa:.4f}")

    # ===== 可视化（bar chart）=====
    plt.figure(figsize=(6,4))

    values = [metrics['all']['mse'], mse_vjepa]
    labels = ["HRM", "Baseline"]

    plt.bar(labels, values)
    plt.ylabel("MSE")
    plt.title("HRM vs Baseline")

    plt.show()


if __name__ == "__main__":
    main()
