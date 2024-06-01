import torch
import torch.nn as nn
import numpy as np
from Networks.networks import *
from Networks.utils import *

class Task_Graph_Encoder(nn.Module):
    def __init__(self, 
                 in_feats=7, 
                 out_feats=256, 
                 edge_feat=2,  
                 hidden_dims=[32,32,96], 
                 k=2, 
                 p=3, 
                 gcn_model='Edge_Laplacian', 
                 activation=nn.ReLU(), 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Task_Graph_Encoder, self).__init__()
        self.linear1 = nn.Linear(in_features=in_feats, out_features=hidden_dims[0])
        self.caps1 = Primary_Capsule(input_dim=hidden_dims[0], 
                                     output_dim=hidden_dims[1], 
                                     k=k, p=p, 
                                     gcn_model=gcn_model, 
                                     activation=activation, 
                                     edge_feat_dim=edge_feat, 
                                     device=device)
        self.caps2 = Secondary_Capsule(input_dim=hidden_dims[1]*p, 
                                       output_dim=hidden_dims[2], 
                                       k=k, p=p, 
                                       gcn_model=gcn_model, 
                                       activation=activation, 
                                       edge_feat_dim=edge_feat, 
                                       device=device)
        self.batch_norm = nn.BatchNorm1d(hidden_dims[2]*p, affine=False, device=device)
        self.linear2 = nn.Linear(in_features=hidden_dims[2]*p, out_features=out_feats, device=device)
        self.final_activation = nn.LeakyReLU()
        self.device = device
        self.to(device)

    def forward(self, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        x = self.linear1(X)
        x_caps1 = self.caps1(X=x, L=L)
        x_caps2 = self.caps2(X=x_caps1, L=L)
        # skip connection
        if  x_caps2.dim() == 2:
            x_caps = x_caps1.repeat(1,int(x_caps2.shape[1]/x_caps1.shape[1])) + x_caps2
        elif x_caps2.dim() == 3:
            x_caps = x_caps1.repeat(1,1,int(x_caps2.shape[2]/x_caps1.shape[2])) + x_caps2
        x_caps = self.batch_norm(x_caps)
        x = self.final_activation(self.linear2(x_caps))
        return x