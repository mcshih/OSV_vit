import torch
import numpy as np
from deit.models_v2 import deit_base_patch16_LS

model = deit_base_patch16_LS()

checkpoint = torch.load('/mnt/HDD1/shih/OSV/deit_osv/pretrain/deit_3_base_224_21k.pth')
model.load_state_dict(checkpoint["model"])

print(model)

