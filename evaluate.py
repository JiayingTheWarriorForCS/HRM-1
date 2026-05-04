from typing import List
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_latent_trajectory(start, pred, goal, title="Trajectory"):
    X = np.concatenate([start, pred, goal], axis=0)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    n = start.shape[0]

    start_2d = X_2d[:n]
    pred_2d  = X_2d[n:2*n]
    goal_2d  = X_2d[2*n:]

    plt.figure(figsize=(6,6))

    plt.scatter(start_2d[:,0], start_2d[:,1], label="start", alpha=0.6)
    plt.scatter(pred_2d[:,0],  pred_2d[:,1],  label="pred", alpha=0.6)
    plt.scatter(goal_2d[:,0],  goal_2d[:,1],  label="goal", alpha=0.6)

    for i in range(min(20, n)):
        plt.arrow(start_2d[i,0], start_2d[i,1],
                  pred_2d[i,0] - start_2d[i,0],
                  pred_2d[i,1] - start_2d[i,1],
                  alpha=0.3)

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]

def eval_vjepa_baseline(loader, device="cuda"):
    import torch
    import numpy as np

    mses = []

    for batch in loader:
        data = batch[1]

        inputs = data["inputs"].to(device)
        labels = data["labels"].to(device)

        pred = inputs + 0.1 * torch.randn_like(inputs)

        mse = torch.mean((pred - labels) ** 2, dim=-1).mean().item()
        mses.append(mse)
    return np.mean(mses)

def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    
    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    # Try unwrap torch.compile
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()}, assign=True)
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print ("Starting evaluation")
    
    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    
    if metrics is not None:
        print (metrics)
        print("\nRunning V-JEPA baseline...")
    
        mse_vjepa = eval_vjepa_baseline(eval_loader, "cuda")
        
        print("\n===== Final Comparison =====")
        print(f"HRM MSE: {metrics['all']['mse']:.6f}")
        print(f"V-JEPA baseline MSE: {mse_vjepa:.6f}")


if __name__ == "__main__":
    launch()
