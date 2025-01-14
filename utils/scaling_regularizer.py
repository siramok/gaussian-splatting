import torch
import torch.nn.functional as F
import math


class ScalingRegularizer:
    def __init__(self, lambda_scaling=1.0, method="default", threshold=450):
        self.lambda_scaling = lambda_scaling
        self.method = method
        self.threshold = threshold

    def compute_loss(self, gaussians):
        if self.method == "default":
            scaling_loss = torch.mean(1.0 / torch.exp(gaussians._scaling))
            return (
                self.lambda_scaling
                * torch.sigmoid(scaling_loss - self.threshold).item()
            )
        elif self.method == "l2_norm":
            scaling_loss = torch.mean(torch.norm(1.0 / gaussians._scaling, p=2))
            return self.lambda_scaling * torch.tanh(scaling_loss)
        elif self.method == "entropy":
            scaling_probs = F.softmax(gaussians._scaling, dim=0)
            entropy_loss = -torch.sum(scaling_probs * torch.log(scaling_probs + 1e-10))
            return self.lambda_scaling * (
                1.0 - entropy_loss / math.log(len(scaling_probs))
            )
        elif self.method == "isqrt":
            scaling_loss = torch.mean(1.0 / torch.sqrt(torch.exp(gaussians._scaling)))
            return self.lambda_scaling * torch.sigmoid(
                scaling_loss - math.sqrt(self.threshold)
            )
        elif self.method == "adaptive":
            mean_scaling = torch.mean(gaussians._scaling)
            scaling_loss = torch.mean(
                1.0 / torch.exp(gaussians._scaling - mean_scaling)
            )
            return (
                self.lambda_scaling * torch.sigmoid(scaling_loss - mean_scaling).item()
            )
