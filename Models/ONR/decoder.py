import torch
import torch.nn as nn
import numpy as np
from Graph_Capsule_Convolution.Networks.networks import *
from Graph_Capsule_Convolution.Networks.utils import *

class MHA_Decoder(nn.Module):
    def __init__(self,
                 context_dim = 128,
                 key_dim = 256,
                 value_dim = 256,
                 num_heads = 8,
                 hidden_dim = [64, 64],
                 out_feats = 50,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MHA_Decoder, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=context_dim,
                                         num_heads=num_heads,
                                         kdim=key_dim,
                                         vdim=value_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=context_dim, affine=False)
        self.linear1 = nn.Linear(in_features=context_dim,
                                 out_features=hidden_dim[0])
        self.linear2 = nn.Linear(in_features=hidden_dim[0],
                                 out_features=hidden_dim[1])
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_dim[1], affine=False)
        self.linear3 = nn.Linear(in_features=hidden_dim[1],
                                 out_features=out_feats)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        self.to(device)

    def forward(self, context, key, value):
        x, _ = self.mha(context, key, value, need_weights=False)
        # x = self.batchnorm1(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        # x = self.batchnorm2(x)
        probs = self.softmax(self.linear3(x))
        return probs