import torch
import numpy as np
from tqdm import tqdm

import yaml
import os
import pydantic
from omegaconf import OmegaConf

from pretrain import PretrainConfig, init_train_state, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str


def evaluate_reasoning(model, loader, device="cuda"):
    model.eval()

    final_mse = []
    intermediate_mse = []
    monotonic_scores = []

    for batch in tqdm(loader):
        start = batch["start_encoding"].to(device)
        f1 = batch["frame_1_encoding"].to(device)
        f2 = batch["frame_2_encoding"].to(device)
        goal = batch["end_encoding"].to(device)

        with torch.no_grad():
            outputs = model(start, goal)
            if isinstance(outputs, dict):
                z1_pred = outputs["z1"]
                z2_pred = outputs["z2"]
            elif isinstance(outputs, tuple):
                z1_pred, z2_pred = outputs
            else:
                raise ValueError("Unknown model output format")

        mse_final = torch.mean((z2_pred - goal) ** 2, dim=-1).mean().item()
        final_mse.append(mse_final)

        mse_f1 = torch.mean((z1_pred - f1) ** 2, dim=-1).mean().item()
        mse_f2 = torch.mean((z2_pred - f2) ** 2, dim=-1).mean().item()
        intermediate_mse.append((mse_f1 + mse_f2) / 2)

        d_start = torch.norm(start - goal, dim=-1)
        d_z1 = torch.norm(z1_pred - goal, dim=-1)
        d_z2 = torch.norm(z2_pred - goal, dim=-1)

        monotonic = ((d_start > d_z1) & (d_z1 > d_z2)).float().mean().item()
        monotonic_scores.append(monotonic)

    return {
        "final_mse": np.mean(final_mse),
        "intermediate_mse": np.mean(intermediate_mse),
        "monotonic_score": np.mean(monotonic_scores),
    }


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== load config =====
    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # ===== dataloader =====
    # _, _ = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1,
    #                         global_batch_size=config.global_batch_size, rank=0, world_size=1)

    # eval_loader, _ = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1,
    #                                    global_batch_size=config.global_batch_size, rank=0, world_size=1)
    # ===== dataloader =====
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    
    eval_loader, _ = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    
    # ===== load model =====
    train_state = init_train_state(config, train_metadata, world_size=1)
    # ===== load model =====
    
    # train_state = init_train_state(config, None, world_size=1)

    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location=device), assign=True)
    except:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location=device).items()},
            assign=True
        )

    model = train_state.model.to(device)

    # ===== run evaluation =====
    print("Running reasoning evaluation...")
    metrics = evaluate_reasoning(model, eval_loader, device)

    print("\n===== HRM Reasoning Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    launch()
