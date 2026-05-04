import torch
import yaml
from pretrain import PretrainConfig, init_train_state, create_dataloader

CKPT_DIR = "/content/drive/MyDrive/HRM_outputs/less_size_run"
CKPT_FILE = "step_14960"

device = "cuda"

# ===== load config =====
with open(f"{CKPT_DIR}/all_config.yaml", "r") as f:
    config = PretrainConfig(**yaml.safe_load(f))

config.checkpoint_path = CKPT_DIR

# ===== dataloader =====
loader, meta = create_dataloader(
    config, "test",
    test_set_mode=True,
    epochs_per_iter=1,
    global_batch_size=config.global_batch_size,
    rank=0, world_size=1
)

# ===== load model =====
state = init_train_state(config, meta, world_size=1)
state.model.load_state_dict(torch.load(f"{CKPT_DIR}/{CKPT_FILE}", map_location=device))
model = state.model.to(device)
model.eval()

# ===== 取一个样本 =====
for _, batch, _ in loader:
    data = {k: v.to(device) for k, v in batch.items()}
    break

# ===== forward（只跑几步，不要等 halt）=====
carry = model.initial_carry(data)

latents = []

for i in range(5):   # ⭐ 关键：拿中间状态
    carry, _, _, _, _ = model(
        carry=carry,
        batch=data,
        return_keys=[]
    )

    # 从 carry 里找 embedding
    for k, v in carry.items():
        if torch.is_tensor(v):
            latents.append(v.detach().cpu())
            print("capture:", k, v.shape)
            break

# 保存
torch.save(latents, "latents.pt")
print("Saved latents!")
