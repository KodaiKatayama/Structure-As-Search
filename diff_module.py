import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#def GCN_diffusion(W, order, feature):
#    """
#    W: [batchsize, n, n]
#    feature: [batchsize, n, n]
#    """
#    identity_matrices = torch.eye(W.size(1)).repeat(W.size(0), 1, 1).to(device)
#    A_gcn = W + identity_matrices
#    degrees = torch.sum(A_gcn, dim=2).unsqueeze(dim=2)
#    D = torch.pow(degrees, -0.5)
#    
#    gcn_diffusion_list = []
#    A_gcn_feature = feature
#    for i in range(order):
#        A_gcn_feature = D * A_gcn_feature
#        A_gcn_feature = torch.matmul(A_gcn, A_gcn_feature)
#        A_gcn_feature = torch.mul(A_gcn_feature, D)
#        gcn_diffusion_list.append(A_gcn_feature)
#    
#    return gcn_diffusion_list


#def GCN_diffusion(W, order, feature):
#    """
#    W: [batchsize, n, n]
#    feature: [batchsize, n, n]
#    """
#    identity_matrices = torch.eye(W.size(1)).repeat(W.size(0), 1, 1).to(device)
#    A_gcn = W + identity_matrices
#    degrees = torch.sum(A_gcn, dim=2).unsqueeze(dim=2)
#    D = torch.pow(degrees, -0.5)
#
#    feature_p = feature
#    gcn_diffusion_list = []
#    for i in range(order):
#        D_feature = D * feature_p
#        A_gcn_D_feature = torch.matmul(A_gcn, D_feature)
#        feature_p = torch.mul(A_gcn_D_feature, D)
#        gcn_diffusion_list.append(feature_p)
#
#    return tuple(gcn_diffusion_list)


def GCN_diffusion(W, order, feature, device):
    """
    Perform GCN diffusion with memory optimization.

    Args:
    W (torch.Tensor): Input tensor of shape [batchsize, n, n]
    order (int): Number of diffusion steps
    feature (torch.Tensor): Feature tensor of shape [batchsize, n, n]
    device (torch.device): The device to perform computations on

    Returns:
    tuple: Tuple of tensors, each of shape [batchsize, n, n]
    """
    #print(W.shape)
    batchsize, n, _ = W.shape

    # Compute A_gcn = W + I efficiently
    A_gcn = W.clone()
    diagonal_indices = torch.arange(n, device=device)
    A_gcn[:, diagonal_indices, diagonal_indices] += 1

    # Compute D^(-1/2) efficiently
    degrees = torch.sum(A_gcn, dim=2, keepdim=True)
    D_inv_sqrt = torch.pow(degrees, -0.5)

    # Normalize A_gcn
    A_gcn_norm = D_inv_sqrt * A_gcn * D_inv_sqrt.transpose(-1, -2)

    # Perform diffusion
    feature_p = feature
    gcn_diffusion_list = []
    for _ in range(order):
        feature_p = torch.bmm(A_gcn_norm, feature_p)
        gcn_diffusion_list.append(feature_p)

    return tuple(gcn_diffusion_list)
#
## Example usage
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#batchsize, n = 32, 100
#W = torch.rand(batchsize, n, n, device=device)
#feature = torch.rand(batchsize, n, n, device=device)
#order = 3
#
#diffusion_results = GCN_diffusion(W, order, feature, device)
#
#print(f"Number of diffusion steps: {len(diffusion_results)}")
#for i, result in enumerate(diffusion_results):
#    print(f"Shape of diffusion result {i}: {result.shape}")
#

#def SCT1stv2(W, order, feature):
#    """
#    W: [batchsize, n, n]
#    feature: [batchsize, n, n]
#    """
#    degrees = torch.sum(W, dim=2).unsqueeze(dim=2)
#    D = torch.pow(degrees, -1)
#    iteration = 2 ** order
#    scale_list = [2 ** i - 1 for i in range(order + 1)]
#    
#    feature_p = feature
#    sct_diffusion_list = []
#    for i in range(iteration):
#        D_inv_x = D * feature_p
#        W_D_inv_x = torch.matmul(W, D_inv_x)
#        feature_p = 0.5 * feature_p + 0.5 * W_D_inv_x
#        if i in scale_list:
#            sct_diffusion_list.append(feature_p)
#    
#    sct_features = [sct_diffusion_list[i] - sct_diffusion_list[i+1] for i in range(len(scale_list) - 1)]
#    
#    return tuple(sct_features)
#


def SCT1stv2(W, order, feature, device):
    """
    Perform SCT diffusion with memory optimization.

    Args:
    W (torch.Tensor): Input tensor of shape [batchsize, n, n]
    order (int): Order of the SCT diffusion
    feature (torch.Tensor): Feature tensor of shape [batchsize, n, n]
    device (torch.device): The device to perform computations on

    Returns:
    tuple: Tuple of tensors, each of shape [batchsize, n, n]
    """
    batchsize, n, _ = W.shape

    # Compute D^(-1) efficiently
    D_inv = torch.pow(W.sum(dim=2, keepdim=True) + 1e-6, -1) #1e-6 for sparse graphs, single node condition

    # Precompute W * D^(-1)
    W_D_inv = W * D_inv.transpose(-1, -2)

    iteration = 2 ** order
    scale_list = [2 ** i - 1 for i in range(order + 1)]

    feature_p = feature
    sct_diffusion_list = []
    for i in range(iteration):
        feature_p = 0.5 * (feature_p + torch.bmm(W_D_inv, feature_p))
        if i in scale_list:
            sct_diffusion_list.append(feature_p)

    sct_features = []
    for i in range(len(scale_list) - 1):
        sct_features.append(sct_diffusion_list[i] - sct_diffusion_list[i+1])
        sct_diffusion_list[i] = None  # Free memory

    return tuple(sct_features)
#
## Example usage
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#batchsize, n = 32, 100
#W = torch.rand(batchsize, n, n, device=device)
#feature = torch.rand(batchsize, n, n, device=device)
#order = 3
#
#sct_results = SCT1stv2(W, order, feature, device)
#
#print(f"Number of SCT features: {len(sct_results)}")
#for i, result in enumerate(sct_results):
#    print(f"Shape of SCT feature {i}: {result.shape}")
#
#
def scattering_diffusion(sptensor, feature,sctorder=7):
    """
    sptensor: [batchsize, n, n]
    feature: [batchsize, n, x]
    """
    h_sct = SCT1stv2(sptensor, sctorder, feature,device)
    return h_sct


if __name__ == '__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchsize, n = 32, 3000
    W = torch.rand(batchsize, n, n, device=device)
    feature = torch.rand(batchsize, n, n, device=device)
    order = 5
    
    sct_results = SCT1stv2(W, order, feature, device)
    
    print(f"Number of SCT features: {len(sct_results)}")
    for i, result in enumerate(sct_results):
        print(f"Shape of SCT feature {i}: {result.shape}")
    
