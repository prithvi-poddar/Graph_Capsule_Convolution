import torch
import torch.nn as nn
import numpy as np
from Graph_Capsule_Convolution.Networks.networks import *
from Graph_Capsule_Convolution.Networks.utils import *


class MHA_Decoder(nn.Module):
    def __init__(
        self,
        context_dim=128,
        key_dim=256,
        value_dim=256,
        num_heads=8,
        hidden_dim=[64, 64],
        out_feats=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(MHA_Decoder, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=value_dim,
            batch_first=True,
        )
        self.layernorm1 = nn.LayerNorm(
            normalized_shape=context_dim
        )  # nn.BatchNorm1d(num_features=context_dim, affine=False)
        self.linear1 = nn.Linear(in_features=context_dim, out_features=hidden_dim[0])
        nn.init.xavier_uniform_(self.linear1.weight.data)
        self.linear2 = nn.Linear(in_features=hidden_dim[0], out_features=out_feats)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        # self.layernorm2 = nn.LayerNorm(normalized_shape=hidden_dim[1])# nn.BatchNorm1d(num_features=hidden_dim[1], affine=False)
        # self.linear3 = nn.Linear(in_features=hidden_dim[1],
        #                          out_features=out_feats)
        # nn.init.xavier_uniform_(self.linear3.weight.data)

        # self.layernorm1 = nn.LayerNorm(normalized_shape=context_dim)# nn.BatchNorm1d(num_features=context_dim, affine=False)
        # self.linear1 = nn.Linear(in_features=context_dim,
        #                          out_features=out_feats)
        # nn.init.xavier_uniform_(self.linear1.weight.data)
        # self.linear2 = nn.Linear(in_features=hidden_dim[0],
        #                          out_features=hidden_dim[1])
        # nn.init.xavier_uniform_(self.linear2.weight.data)
        # self.layernorm2 = nn.LayerNorm(normalized_shape=hidden_dim[1])# nn.BatchNorm1d(num_features=hidden_dim[1], affine=False)
        # self.linear3 = nn.Linear(in_features=hidden_dim[1],
        #                          out_features=out_feats)
        # nn.init.xavier_uniform_(self.linear3.weight.data)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        self.to(device)

    def forward(self, context, key, value):
        # if len(context.shape)==2:
        #     key = key.squeeze(0)
        #     value = value.squeeze(0)
        x, _ = self.mha(context, key, value, need_weights=False)
        x = self.layernorm1(x)
        x = self.activation(self.linear1(x))
        # x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        # # x = self.layernorm2(x)
        # probs = self.softmax(self.linear3(x))
        return x  # probs


class SATA_Decoder(nn.Module):
    def __init__(
        self,
        decoder_input_dims=128,
        decoder_hidden_dim=[64, 64],
        decoder_out_feats=50,
        use_batchnorm=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(SATA_Decoder, self).__init__()
        layers = []

        layers.append(nn.Linear(decoder_input_dims, decoder_hidden_dim[0]))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(decoder_hidden_dim[0]))

        layers.append(nn.ReLU())

        for i in range(1, len(decoder_hidden_dim)):
            layers.append(nn.Linear(decoder_hidden_dim[i - 1], decoder_hidden_dim[i]))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(decoder_hidden_dim[i]))

            layers.append(nn.ReLU())

        layers.append(nn.Linear(decoder_hidden_dim[-1], decoder_out_feats))

        self.model = nn.Sequential(*layers)

        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.model(x)
        return x  # probs
