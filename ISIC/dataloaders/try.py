import torch
import torch.nn.functional as F
import numpy as np


ema_preds, ema_sdf_probs, ema_recon_probs = torch.zeros([3, 8, 4, 2, 256, 256]).cuda()
# print(ema_preds.size())

# ema_preds = F.softmax(ema_preds, dim=2)
# uncertainty = -1.0 * torch.sum(ema_preds * torch.log2(ema_preds + 1e-6), dim=2,
# 							   keepdim=True)
# weights = F.softmax(1 - uncertainty, dim=0)
# print(weights.size())
# ema_probs = torch.sum(ema_preds * weights, dim=0)
# print(ema_probs.size())

a = torch.Tensor([1, 2])
b = torch.Tensor([3, 4])
lis = [a, b]
# print(lis)
# c = torch.Tensor([5, 6])
# d = torch.Tensor([7, 8])
# lis1 = [c, d]
# print(lis1)
# res = np.multiply(lis,lis1)
# print(res)
a = [1,2]
# print(sum(a))
import torch

print(torch.__version__)  # 注意是双下划线