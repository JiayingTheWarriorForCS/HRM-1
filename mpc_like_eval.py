import torch
import numpy as np

latents = torch.load("latents.pt")

# 假设最后一个是 goal embedding（近似）
goal = latents[-1]

for i, z in enumerate(latents):
    dist = torch.norm(z - goal)
    print(f"step {i}: distance to goal = {dist.item():.4f}")
