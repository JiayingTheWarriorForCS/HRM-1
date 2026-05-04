import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

PRED_PATH = "/content/drive/MyDrive/HRM_outputs/less_size_run/step_14960_all_preds.0"


def plot_latent_trajectory(start, pred, goal, title):
    X = np.concatenate([start, pred, goal], axis=0)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    n = start.shape[0]

    s = X_2d[:n]
    p = X_2d[n:2*n]
    g = X_2d[2*n:]

    plt.figure(figsize=(6,6))
    plt.scatter(s[:,0], s[:,1], label="start", alpha=0.6)
    plt.scatter(p[:,0], p[:,1], label="pred", alpha=0.6)
    plt.scatter(g[:,0], g[:,1], label="goal", alpha=0.6)

    for i in range(min(20, n)):
        plt.arrow(s[i,0], s[i,1],
                  p[i,0] - s[i,0],
                  p[i,1] - s[i,1],
                  alpha=0.3)

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def main():
    print("Loading predictions...")

    all_preds = torch.load(PRED_PATH)

    print("Keys:", all_preds.keys())

    inputs = all_preds["inputs"].numpy()
    labels = all_preds["labels"].numpy()

    L = labels.shape[1]

    start = labels[:, :L//4]
    goal  = labels[:, L//4 : L//2]
    pred_hrm = inputs[:, :L//4]

    plot_latent_trajectory(start, pred_hrm, goal, "HRM")

    plot_latent_trajectory(start, start, goal, "V-JEPA baseline")


if __name__ == "__main__":
    main()
