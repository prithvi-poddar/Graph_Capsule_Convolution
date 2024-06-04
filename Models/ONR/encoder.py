import torch
import torch.nn as nn
import numpy as np
from Graph_Capsule_Convolution.Networks.networks import *
from Graph_Capsule_Convolution.Networks.utils import *
# from Networks.networks import *
# from Networks.utils import *

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
    
class Context_Encoder(nn.Module):
    def __init__(self, 
                 in_feats = 8,
                 out_feats = 128,
                 hidden_dims=[32,64],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Context_Encoder, self).__init__()
        self.agent_linear1 = nn.Linear(in_features=in_feats,
                                       out_features=hidden_dims[0])
        self.agent_linear2 = nn.Linear(in_features=hidden_dims[0],
                                       out_features=hidden_dims[1])
        self.linear1 = nn.Linear(in_features=in_feats,
                                 out_features=hidden_dims[0])
        self.linear2 = nn.Linear(in_features=hidden_dims[0],
                                 out_features=hidden_dims[1])
        self.context = nn.Linear(in_features=hidden_dims[1]*2+1,
                                 out_features=out_feats)
        self.activation = nn.ReLU()
        self.device = device
        self.to(device)

    def forward(self, agent, peers, time):
        ag_x = self.activation(self.agent_linear1(agent))
        ag_x_1 = self.activation(self.agent_linear2(ag_x))
        # skip connection
        if  ag_x_1.dim() == 2:
            ag_x_2 = ag_x.repeat(1,int(ag_x_1.shape[1]/ag_x.shape[1])) + ag_x_1
        elif ag_x_1.dim() == 3:
            ag_x_2 = ag_x.repeat(1,1,int(ag_x_1.shape[2]/ag_x.shape[2])) + ag_x_1
        
        peer_x = self.activation(self.linear1(peers))
        peer_x_1 = self.activation(self.linear2(peer_x))
        # skip connection
        if  peer_x_1.dim() == 2:
            peer_x_2 = peer_x.repeat(1,int(peer_x_1.shape[1]/peer_x.shape[1])) + peer_x_1
        elif peer_x_1.dim() == 3:
            peer_x_2 = peer_x.repeat(1,1,int(peer_x_1.shape[2]/peer_x.shape[2])) + peer_x_1
        # TODO: Potential failure point
        peer_x_2 = torch.sum(peer_x_2, dim=0, keepdim=True)
        x = self.activation(self.context(torch.cat((time, ag_x_2, peer_x_2), dim=1)))
        return x