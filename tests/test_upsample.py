import torch
from torch import nn

im = torch.zeros((3, 3, 5, 5))
print(im.shape)
model = nn.Upsample(2)
im = model(im)
print(im.shape)
im = model(im)
print(im.shape)
