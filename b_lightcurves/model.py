"""Small 1D ResNet: stem conv k=7 stride 2, 4 stages (32, 64, 128, 128) of 2 residual blocks each,
stride 2 at the start of every stage, masked global average pool to a 128-d embedding, linear head.

The pool averages over observed positions only (mask channel), so a 30 s input is not diluted by
970 s of zero padding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, n_classes, widths=(32, 64, 128, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, widths[0], 7, 2, 3, bias=False), nn.BatchNorm1d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for w in widths:
            layers += [Block(cin, w, 2), Block(w, w, 1)]
            cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], n_classes)

    def embed(self, x):
        h = self.body(self.stem(x))
        mask = F.adaptive_max_pool1d(x[:, 1:2], h.shape[-1])
        return (h * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    def forward(self, x):
        e = self.embed(x)
        return self.head(e), e
