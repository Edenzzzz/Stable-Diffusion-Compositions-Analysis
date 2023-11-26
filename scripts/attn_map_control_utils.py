import torch
from torch.utils.data import Dataset

class sd_model_wrapper():
    """
    A wrapper that handles both the huggingface Stable Diffusion pipeline
    and the and the heavier version in the original repo.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def __call__(self, x):
        return self.model(x)