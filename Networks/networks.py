import torch 
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.functional as F
from typing import List
from torch_geometric.utils import get_laplacian

class Laplacian_GCN_parallel(nn.Module):
    # TODO: check the backpropogation
    def __init__(self, input_dim, output_dim, k, activation=nn.ReLU(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Laplacian_GCN_parallel, self).__init__()
        self.__in_dim = input_dim
        self.__out_dim = output_dim
        self.__k = k
        self.act = activation
        self.weights = nn.ModuleList([nn.Linear(self.__in_dim, self.__out_dim) for i in range(self.__k+1)])
        self.device = device
        self.to(device)

    def conv(self, X, L, W, k):
        return W(torch.matmul(torch.linalg.matrix_power(L,k),X))
    
    def forward(self, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        futures : List[torch.jit.Future[torch.Tensor]] = []
        for k in range(self.__k+1):
            futures.append(torch.jit.fork(self.conv, X, L, self.weights[k], k))
        
        results = torch.empty((self.__k+1, X.shape[0], self.__out_dim))
        for idx, future in enumerate(futures):
            results[idx,:,:] = torch.jit.wait(future)
        X = torch.sum(results, dim=0)
        return self.act(X)
    
class Laplacian_GCN(nn.Module):
    def __init__(self, input_dim, output_dim, k, activation=nn.ReLU(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Laplacian_GCN, self).__init__()
        self.__in_dim = input_dim
        self.__out_dim = output_dim
        self.__k = k
        self.act = activation
        self.weights = nn.ModuleList([nn.Linear(self.__in_dim, self.__out_dim) for i in range(self.__k+1)])
        for idx, net in enumerate(self.weights):
            nn.init.xavier_uniform_(net.weight.data)
        self.device = device
        self.to(device)
    
    def forward(self, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        X_sum = torch.zeros((X.shape[0], self.__out_dim), device=self.device, requires_grad=True)
        for k in range(self.__k+1):
            X_sum = X_sum + self.weights[k](torch.matmul(torch.linalg.matrix_power(L,k),X))
        return self.act(X_sum)  

class Edge_Laplacian_GCN(nn.Module):
    def __init__(self, input_dim, edge_input_dim, output_dim, k, activation=nn.ReLU(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Edge_Laplacian_GCN, self).__init__()
        self.__in_dim = input_dim
        self.__edge_in_dim = edge_input_dim
        self.__out_dim = output_dim

        assert output_dim%edge_input_dim == 0, "output dim not divisible by edge input dim"

        self.__k = k
        self.act = activation
        self.device = device

        self.nets = [Laplacian_GCN(input_dim=self.__in_dim, output_dim=int(self.__out_dim/self.__edge_in_dim), k=self.__k, activation=self.act, device=self.device) for i in range(self.__edge_in_dim)]
        # self.weights = nn.ModuleList([nn.Linear(self.__in_dim, self.__out_dim) for i in range(self.__k+1)])
        self.to(device)
    
    def forward(self, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        E = [L[:,:,i] for i in range(L.shape[2])]
        x = []
        for idx, net in enumerate(self.nets):
            x.append(net(X=X, L=E[idx]))
        X_final = torch.cat(x, axis=1)
        return X_final  
    
class Primary_Capsule(nn.Module):
    #TODO: Do we take the statistical moments in the first layer?
    def __init__(self, input_dim, output_dim, gcn_model='Edge_Laplacian', p=3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super(Primary_Capsule, self).__init__()
        self.__in_dim = input_dim
        self.__out_dim = output_dim
        self.p = p
        self.activation = kwargs['activation']
        self.device = device
        if gcn_model == 'Laplacian':
            k = kwargs['k']
            activation = kwargs['activation']
            self.nets = nn.ModuleList([Laplacian_GCN(input_dim=self.__in_dim, output_dim=self.__out_dim, k=k, activation=activation, device=self.device) for i in range(self.p)])
        elif gcn_model == 'Edge_Laplacian':
            k = kwargs['k']
            activation = kwargs['activation']
            edge_feat_dim = kwargs['edge_feat_dim']
            self.nets = nn.ModuleList([Edge_Laplacian_GCN(input_dim=self.__in_dim, edge_input_dim=edge_feat_dim, output_dim=self.__out_dim, k=k, activation=activation, device=self.device) for i in range(self.p)])
        self.batchnorm = nn.BatchNorm1d(num_features=self.__out_dim*self.p)
        

    def forward(self, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        outs = []
        for idx, net in enumerate(self.nets):
            outs.append(net(X=X, L=L))
        output = torch.stack(outs, -1)
        output = self.batchnorm(torch.flatten(output, -2, -1))
        return self.activation(output)
    
class Secondary_Capsule(nn.Module):
    def __init__(self, input_dim, output_dim, gcn_model='Edge_Laplacian', p=3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super(Secondary_Capsule, self).__init__()
        self.__in_dim = input_dim
        self.__out_dim = output_dim
        self.p = p
        self.activation = kwargs['activation']
        self.device = device
        if gcn_model == 'Laplacian':
            k = kwargs['k']
            activation = kwargs['activation']
            self.nets = nn.ModuleList([Laplacian_GCN(input_dim=self.__in_dim, output_dim=self.__out_dim, k=k, activation=activation, device=self.device) for i in range(self.p)])
        elif gcn_model == 'Edge_Laplacian':
            k = kwargs['k']
            activation = kwargs['activation']
            edge_feat_dim = kwargs['edge_feat_dim']
            self.nets = nn.ModuleList([Edge_Laplacian_GCN(input_dim=self.__in_dim, edge_input_dim=edge_feat_dim, output_dim=self.__out_dim, k=k, activation=activation, device=self.device) for i in range(self.p)])
        self.batchnorm = nn.BatchNorm1d(num_features=self.__out_dim*self.p)
        
        
    def forward(self, data=None, X=None, L=None):
        if X == None:
            X = data.x
            L = get_laplacian(data.edge_index)
        X_moments = X*X
        for _ in range(self.p-2):
            X_moments = X_moments*X
        outs = []
        for idx, net in enumerate(self.nets):
            outs.append(net(X=X_moments, L=L))
        output = torch.stack(outs, -1)
        output = self.batchnorm(torch.flatten(output, -2, -1))
        return self.activation(output)