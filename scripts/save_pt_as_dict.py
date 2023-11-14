import torch
from torch import nn
from pathlib import Path

file = Path("tmp/yolov8n.pt")

model = torch.load(file)
model: nn.Module = model["model"]
state_dict = model.float().state_dict()
torch.save(state_dict, file.with_name(file.stem + "-dict.pt"))
