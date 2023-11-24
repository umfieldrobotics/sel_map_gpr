import numpy as np
import torch


class BypassWrapper():
    def __init__(self, model=None, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass
    
    def runSegmentation(self, pil_image, return_numpy=True, one_hot=False):
        width, height = pil_image.size
        if return_numpy:
            return np.zeros((0, height, width)), 1 # confidence is 1 when there's no semantics
        else:
            return torch.zeros((0, height, width), device=self.device), 1 # confidence is 1 when there's no semantics
