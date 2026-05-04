import torch
import numpy as np
import matplotlib.pyplot as plt

PRED_PATH = "/content/drive/MyDrive/HRM_outputs/less_size_run/step_14960_all_preds.0"


def compute_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


def main():
    print("Loading predictions...")
    data = torch.load(PRED_PATH)

    inputs = data["inputs"].numpy()
    labels = data["labels"].numpy()

    L = labels.shape[1]

    start = labels[:, :L//4]         
    goal  = labels[:, L//4 : L//2]   

    pred_hrm = inputs[:, :L//4]

    pred_vjepa = start

    start = start.mean(axis=1)
    goal  = goal.mean(axis=1)
    pred_hrm = pred_hrm.mean(axis=1)
    pred_vjepa = pred_vjepa.mean(axis=1)

    d_start = compute_distance(start, goal)
    d_hrm   = compute_distance(pred_hrm, goal)
    d_vjepa = compute_distance(pred_vjepa, goal)

    idx = np.argsort(d_start)

    d_start = d_start[idx]
    d_hrm   = d_hrm[idx]
    d_vjepa = d_vjepa[idx]

    plt.figure(figsize=(8,5))

    plt.plot(d_start, label="Start → Goal", linestyle="--")
    plt.plot(d_hrm, label="HRM → Goal")
    plt.plot(d_vjepa, label="V-JEPA → Goal")

    plt.xlabel("Sample Index (sorted)")
    plt.ylabel("Distance to Goal")
    plt.title("HRM vs V-JEPA Distance Comparison")

    plt.legend()
    plt.grid()

    plt.show()

    print("\n===== Summary =====")
    print(f"Start → Goal: {np.mean(d_start):.4f}")
    print(f"HRM   → Goal: {np.mean(d_hrm):.4f}")
    print(f"V-JEPA→ Goal: {np.mean(d_vjepa):.4f}")

    print("\n===== Improvement =====")
    print(f"HRM improvement: {(np.mean(d_start) - np.mean(d_hrm)):.4f}")
    print(f"V-JEPA improvement: {(np.mean(d_start) - np.mean(d_vjepa)):.4f}")


if __name__ == "__main__":
    main()
