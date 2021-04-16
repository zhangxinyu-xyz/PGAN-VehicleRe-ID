import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class PAM(nn.Module):
    def __init__(self, attention_dim, reduction=4, temp=10.0):
        super(PAM, self).__init__()

        self.temp = temp
        self.linear1 = weight_norm(nn.Linear(attention_dim, attention_dim // reduction, bias=False), dim=None)
        self.ln1 = nn.LayerNorm(attention_dim // reduction)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = weight_norm(nn.Linear(attention_dim // reduction, attention_dim, bias=False), dim=None)

    def forward(self, v):
        """
        v: [batch, k, attention_dim]
        """
        v = self.logits(v)
        v = v.mean(2, keepdim=True)   # --> [b, p, 1]
        v = nn.functional.softmax(v * self.temp, 1)
        return v.unsqueeze(-1)

    def logits(self, v):
        v = self.linear1(v)
        v = self.ln1(v)
        v = self.relu1(v)
        v = self.linear2(v)
        return v


