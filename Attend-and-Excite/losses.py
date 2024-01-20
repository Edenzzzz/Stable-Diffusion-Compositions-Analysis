import torch
import torch.nn.functional as F

def Distance_Correlation(latent, control):
    """ Distance correlation from https://github.com/zhenxingjian/Partial_Distance_Correlation
    Measures the similarity between embedding spaces

    Args:
        latent (torch.Tensor): (N, D): tensor of latent embeddings
        control (torch.Tensor): (N, D): tensor of control embeddings
    """
    latent = torch.atleast_2d(latent)
    control = torch.atleast_2d(control)
    
    latent = F.normalize(latent)
    control = F.normalize(control)

    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

    matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

    
    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    # correlation_r = torch.pow(Gamma_XY,2)/(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r

@torch.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=False)
def matrix_sqrt(mat):
    mat = mat.float()
    U, S, V = torch.linalg.svd(mat)
    sqrt_S = torch.sqrt(S)
    return U @ torch.diag(sqrt_S) @ V.T


def Wasserstein_loss(x, y):
    """
    Wasserstein distance between two multivariate Gaussians
    Args:
        x (torch.Tensor): (N, D): tensor of latent embeddings
        y (torch.Tensor): (N, D): tensor of control embeddings

    """
    x = torch.atleast_2d(x)
    y = torch.atleast_2d(y)

    # Estimate means
    mean_x = torch.mean(x, dim=0)
    mean_y = torch.mean(y, dim=0)

    # Estimate covariance matrices
    cov_x = torch.cov(x.T, correction=0)
    cov_y = torch.cov(y.T, correction=0)
    # avoid singular matrix
    reg = 1e-6
    cov_x += reg * torch.eye(cov_x.shape[0], device=cov_x.device)
    cov_y += reg * torch.eye(cov_y.shape[0], device=cov_y.device)
    product = cov_x @ cov_y
    if product.isnan().any():
        return torch.tensor(0.0)
    
    # Compute mean difference squared
    mean_diff_squared = torch.norm(mean_x - mean_y, p=2) ** 2

    # Compute the trace term 
    sqrt_cov_product = matrix_sqrt(product)
    if not isinstance(sqrt_cov_product, torch.Tensor):
        sqrt_cov_product = sqrt_cov_product.real  # Handling complex eigenvalues
    trace_term = torch.trace(cov_x + cov_y - 2 * sqrt_cov_product)

    # Wasserstein distance
    return mean_diff_squared + trace_term

