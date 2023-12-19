import torch

def Distance_Correlation(latent, control):

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


def Wasserstein_loss(x, y):
    # Ensure x and y have the same number of samples
    assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"

    # Estimate means
    mean_x = torch.mean(x, dim=0)
    mean_y = torch.mean(y, dim=0)

    # Estimate covariance matrices
    cov_x = torch.cov(x.T)
    cov_y = torch.cov(y.T)

    # Compute mean difference squared
    mean_diff_squared = torch.norm(mean_x - mean_y, p=2)**2

    # Compute the trace term
    sqrt_cov_product = torch.sqrtm(cov_x @ cov_y)
    if not isinstance(sqrt_cov_product, torch.Tensor):
        sqrt_cov_product = sqrt_cov_product.real  # Handling complex eigenvalues
    trace_term = torch.trace(cov_x + cov_y - 2 * sqrt_cov_product)

    # Wasserstein distance
    return mean_diff_squared + trace_term

