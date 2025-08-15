import torch
import torch.nn.functional as F
from torch import nn
from diff_module import scattering_diffusion, GCN_diffusion  # Ensure these are correctly imported
#from sinkhorn import  gumbel_sinkhorn


@torch.compile
def fast_sinkhorn_step(Z):
    # Row normalization
    Z = Z - torch.logsumexp(Z, dim=-1, keepdim=True)
    # Column normalization
    Z = Z - torch.logsumexp(Z, dim=-2, keepdim=True)
    return Z


def gumbel_sinkhorn(log_alpha, tau=1.0, n_iter=20, noise_scale=1.0):
    # ... initialization ...
    B, N, _ = log_alpha.size()
    
    # Add Gumbel noise
    noise = -torch.log(-torch.log(torch.rand(log_alpha.shape, device=log_alpha.device) + 1e-20) + 1e-20)
    noise = noise * noise_scale
    
    Z = (log_alpha + noise) / tau    
    for _ in range(n_iter):
        Z = fast_sinkhorn_step(Z)
    
    P = torch.exp(Z)
    # Ensure exact normalization
    P = P / P.sum(dim=-1, keepdim=True)
    P = P / P.sum(dim=-2, keepdim=True)
    return P, (log_alpha + noise) / tau




#def gumbel_sinkhorn(log_alpha, tau=1.0, n_iter=20,noise_scale=1.0):
#    """Stable Gumbel-Sinkhorn with better scaling"""
#    B, N, _ = log_alpha.size()
#
#    # Scale and center the input logits
#    # log_alpha = log_alpha * 0.01  # Scale down
#    # log_alpha = log_alpha - log_alpha.mean(dim=(1,2), keepdim=True)
#
#    # Add Gumbel noise
#    noise = -torch.log(-torch.log(torch.rand(log_alpha.shape, device=log_alpha.device) + 1e-20) + 1e-20)
#    noise = noise*noise_scale   # Scale noise too
#
#    # Initialize the iteration
#    Zinit = (log_alpha + noise) / tau
#    Z = Zinit
#
#    for _ in range(n_iter):
#        # Row normalization
#        Z = Z - torch.logsumexp(Z, dim=-1, keepdim=True)
#        # Column normalization
#        Z = Z - torch.logsumexp(Z, dim=-2, keepdim=True)
#
#    # Get probabilities
#    P = torch.exp(Z)
#
#    # Ensure exact normalization
#    P = P / P.sum(dim=-1, keepdim=True)
#    P = P / P.sum(dim=-2, keepdim=True)
#
#    return P,Zinit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SCTConv(nn.Module):
    def __init__(self, hidden_dim, sctorder=2,gcnorder=2):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        self.sctorder = sctorder
        self.gcnorder = gcnorder
        self.num_of_chnl = sctorder + gcnorder  # 2 GCN + scattering order
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    def forward(self, X, adj, moment=1):
        support0 = X
        B = support0.size(0)  # batchsize
        N = support0.size(1)  # number of nodes

        # Normalize adjacency matrix----
        # adj = F.normalize(adj, p=1, dim=2)
        # GCN diffusion
        gcn_diffusion_list = GCN_diffusion(adj, self.gcnorder, support0,device)
        gcn_diffusion_list = [nn.LeakyReLU()(gcn_diff) for gcn_diff in gcn_diffusion_list]

        # Scattering diffusion
        h_sct_list = scattering_diffusion(adj, support0,self.sctorder)

        # Apply moment to scattering features
        h_sct_list = [torch.abs(h_sct) ** moment for h_sct in h_sct_list]

        # Concatenate GCN and scattering diffusion features
        a_input_list = [torch.cat((support0, gcn_diff), dim=2).unsqueeze(1) for gcn_diff in gcn_diffusion_list] + \
                       [torch.cat((support0, h_sct), dim=2).unsqueeze(1) for h_sct in h_sct_list]
        a_input = torch.cat(a_input_list, dim=1).view(B, self.num_of_chnl, N, -1)
        e = torch.matmul(F.relu(a_input), self.a).squeeze(3)

        attention = F.softmax(e+ 1e-10, dim=1).view(B, self.num_of_chnl, N, -1)
        h_all = torch.cat([gcn_diff.unsqueeze(dim=1) for gcn_diff in gcn_diffusion_list] + \
                          [h_sct.unsqueeze(dim=1) for h_sct in h_sct_list], dim=1).view(B, self.num_of_chnl, N, -1)

        h_prime = torch.mul(attention, h_all)
        h_prime = torch.mean(h_prime, 1)  # (B, n, f)
        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, sctorder=8,gcnorder=1,TanhScale=40,tau=0.1, n_iter=20,noise_scale=0.05,Inference=False):
        super().__init__()
        self.input_dim = input_dim
        self.TanhScale = TanhScale
        self.tau = tau
        self.n_iter = n_iter
        self.noise_scale = noise_scale
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.inference = Inference
        for _ in range(n_layers):
            self.convs.append(SCTConv(hidden_dim, sctorder,gcnorder))


        print('Total number of sct channels: %d'%sctorder)
        print('Total number of low pass channels: %d'%gcnorder)




        self.mlp1 = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.m = nn.Softmax(dim=1)
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.init.xavier_uniform_(self.mlp2.weight)
        nn.init.xavier_uniform_(self.in_proj.weight)

    def forward(self, X, adj, moment=1):
        X = self.in_proj(X)
        hidden_states = X
        for layer in self.convs:
            X = layer(X, adj, moment=moment)
            hidden_states = torch.cat([hidden_states, X], dim=-1)

        X = hidden_states
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        logits = self.mlp2(X)
        logits = torch.tanh(logits) * self.TanhScale  # This will bound them to [-40, 40]
        if not self.inference:
            X,gsinit = gumbel_sinkhorn(logits, tau=self.tau, n_iter=self.n_iter,noise_scale=self.noise_scale)
            return X, gsinit
        else:
            # Add Gumbel noise
            noise = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device) + 1e-20) + 1e-20)
            noise = noise*self.noise_scale   # Scale noise too
            Zinit = (logits + noise) / self.tau
            return Zinit

        
        
