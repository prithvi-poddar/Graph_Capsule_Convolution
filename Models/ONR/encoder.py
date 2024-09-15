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
        self.batchnorm1 = nn.LayerNorm(hidden_dims[0], elementwise_affine=True)#nn.BatchNorm1d(hidden_dims[0], affine=True)

        self.caps1 = Primary_Capsule(input_dim=hidden_dims[0], 
                                     output_dim=hidden_dims[1], 
                                     k=k, p=p, 
                                     gcn_model=gcn_model, 
                                     activation=activation, 
                                     edge_feat_dim=edge_feat, 
                                     device=device)
        self.caps_1_norm = nn.LayerNorm(hidden_dims[1]*p, elementwise_affine=True)# nn.BatchNorm1d(hidden_dims[1]*p, affine=True)

        self.caps2 = Secondary_Capsule(input_dim=hidden_dims[1]*p, 
                                       output_dim=hidden_dims[2], 
                                       k=k, p=p, 
                                       gcn_model=gcn_model, 
                                       activation=activation, 
                                       edge_feat_dim=edge_feat, 
                                       device=device)
        self.caps_2_norm = nn.LayerNorm(hidden_dims[2]*p, elementwise_affine=True)# nn.BatchNorm1d(hidden_dims[2]*p, affine=True)

        self.caps_norm = nn.LayerNorm(hidden_dims[2]*p, elementwise_affine=True) #nn.BatchNorm1d(hidden_dims[2]*p, elementwise_affine=True)
        # self.batch_norm = nn.BatchNorm1d(100, affine=False, device=device)
        
        self.linear2 = nn.Linear(in_features=hidden_dims[2]*p, out_features=out_feats, device=device)
        self.batchnorm2 = nn.LayerNorm(out_feats, elementwise_affine=True)# nn.BatchNorm1d(out_feats, affine=False)

        self.final_activation = nn.LeakyReLU()
        self.device = device
        self.to(device)

    def forward(self, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        
        x = self.linear1(X)
        x = self.batchnorm1(x)
        # n_batch, num_nodes, l1_dim = x.shape
        # x_reshaped = x.view(-1, l1_dim)
        # x_normalized = self.batchnorm1(x_reshaped)
        # x = x_normalized.view(n_batch, num_nodes, l1_dim)

        # x = self.batchnorm1(self.linear1(X))
        x_caps1 = self.caps1(X=x, L=L)
        x_caps1 = self.caps_1_norm(x_caps1)

        # n_batch, num_nodes, caps_1_dim = x_caps1.shape
        # x_reshaped = x_caps1.view(-1, caps_1_dim)
        # x_normalized = self.caps_1_norm(x_reshaped)
        # x_caps1 = x_normalized.view(n_batch, num_nodes, caps_1_dim)


        x_caps2 = self.caps2(X=x_caps1, L=L)
        x_caps2 = self.caps_2_norm(x_caps2)
        # n_batch, num_nodes, caps_2_dim = x_caps2.shape
        # x_reshaped = x_caps2.view(-1, caps_2_dim)
        # x_normalized = self.caps_2_norm(x_reshaped)
        # x_caps2 = x_normalized.view(n_batch, num_nodes, caps_2_dim)

        # skip connection
        if  x_caps2.dim() == 2:
            x_caps = x_caps1.repeat(1,int(x_caps2.shape[1]/x_caps1.shape[1])) + x_caps2
        elif x_caps2.dim() == 3:
            x_caps = x_caps1.repeat(1,1,int(x_caps2.shape[2]/x_caps1.shape[2])) + x_caps2

        # n_batch, num_nodes, caps_dim = x_caps.shape
        # x_reshaped = x_caps.view(-1, caps_dim)
        # x_normalized = self.caps_norm(x_reshaped)
        # x_caps = x_normalized.view(n_batch, num_nodes, caps_dim)
        
        x = self.final_activation(self.linear2(x_caps))
        # n_batch, num_nodes, x_dim = x.shape
        # x_reshaped = x.view(-1, x_dim)
        # x_normalized = self.batchnorm2(x_reshaped)
        # x = x_normalized.view(n_batch, num_nodes, x_dim)
       
        return x
    
class Context_Encoder_V2(nn.Module):
    def __init__(self, 
                 in_feats = 7,
                 out_feats = 128,
                 edge_feat=1,
                 hidden_dims=[32,32,32],
                 k=2, 
                 p=3, 
                 gcn_model='Laplacian', 
                 activation=nn.ReLU(),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Context_Encoder_V2, self).__init__()
        self.linear1 = nn.Linear(in_features=in_feats, out_features=hidden_dims[0])
        self.batchnorm1 = nn.LayerNorm(hidden_dims[0], elementwise_affine=True)

        self.caps1 = Primary_Capsule(input_dim=hidden_dims[0], 
                                     output_dim=hidden_dims[1], 
                                     k=k, p=p, 
                                     gcn_model=gcn_model, 
                                     activation=activation, 
                                     edge_feat_dim=edge_feat, 
                                     device=device)
        self.caps_1_norm = nn.LayerNorm(hidden_dims[1]*p, elementwise_affine=True)

        self.caps2 = Secondary_Capsule(input_dim=hidden_dims[1]*p, 
                                       output_dim=hidden_dims[2], 
                                       k=k, p=p, 
                                       gcn_model=gcn_model, 
                                       activation=activation, 
                                       edge_feat_dim=edge_feat, 
                                       device=device)
        self.caps_2_norm = nn.LayerNorm(hidden_dims[2]*p, elementwise_affine=True)
        self.caps_norm = nn.LayerNorm(hidden_dims[2]*p, elementwise_affine=True)

        # self.batch_norm = nn.BatchNorm1d(hidden_dims[2]*p, affine=False, device=device)
        self.context_l1 = nn.Linear(in_features=hidden_dims[2]*p*2+1,
                                 out_features=hidden_dims[2]*p)
        self.context_l2 = nn.Linear(in_features=hidden_dims[2]*p,
                                 out_features=out_feats)
        self.batchnorm2 = nn.LayerNorm(out_feats, elementwise_affine=True)

        self.activation = nn.LeakyReLU()
        self.device = device
        self.to(device)

    def forward(self, agent_idx, time, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        x = self.linear1(X)
        x = self.batchnorm1(x)

        x_caps1 = self.caps1(X=x, L=L)
        x_caps1 = self.caps_1_norm(x_caps1)

        x_caps2 = self.caps2(X=x_caps1, L=L)
        x_caps2 = self.caps_2_norm(x_caps2)
        # skip connection
        if  x_caps2.dim() == 2:
            x_caps = x_caps1.repeat(1,int(x_caps2.shape[1]/x_caps1.shape[1])) + x_caps2
        elif x_caps2.dim() == 3:
            x_caps = x_caps1.repeat(1,1,int(x_caps2.shape[2]/x_caps1.shape[2])) + x_caps2
        # x_caps = self.batch_norm(x_caps)

        batch_indices = torch.arange(agent_idx.size(0)).unsqueeze(1)

        agent_x = x_caps[batch_indices, agent_idx]

        agent_idx_ = agent_idx.squeeze(0)#.squeeze(1)
        rows = torch.arange(x_caps.shape[1])  # This gives [0, 1, 2]

        mask = torch.ones_like(x_caps[:, :, 0], dtype=torch.bool)
        for i in range(x_caps.shape[0]):
            mask[i, agent_idx_[i]] = False  # Set False for rows indexed by b

        # Index the tensor with the mask to get rows apart from those in b
        peers_x = x_caps[mask].reshape(x_caps.shape[0], -1, x_caps.shape[2])

        # agent_x = x_caps[torch.arange(x_caps.shape[0]), agent_idx.squeeze()] #x_caps[:,agent_idx,:]
        # peers_x = torch.cat((x_caps[:,:agent_idx,:],x_caps[:,agent_idx+1:,:]), dim=1)
        peers_x = torch.sum(peers_x, dim=1, keepdim=True)
        # if len(time.shape) == 2:
        #     time = time.unsqueeze(0)
        # time = torch.unsqueeze(time, dim=0)
        # print(time)

        x = self.activation(self.context_l1(torch.cat((time, agent_x.squeeze(1), peers_x.squeeze(1)), dim=1)))
        x = self.context_l2(x)
        x = self.activation(self.batchnorm2(x))

        # x = self.activation(self.context(torch.cat((time, agent_x.squeeze(1), peers_x.squeeze(1)), dim=1)))
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
    
