import torch
from typing import Literal

class SimilarityMetric:
    def __init__(
        self,
        metric_type: Literal["L1", "L2", "DOT", "COS", "ATAN_DOT", "TANH_DOT", "SQRT_DOT"]
    ):
        if metric_type not in SimilarityMetric.implementation_table:
            raise ValueError("Invalid Metric Type")
        
        self.metric_type = metric_type

    def __call__(self, x: torch.Tensor):
        return SimilarityMetric.implementation_table[self.metric_type](x)
    
    @staticmethod
    def l1(x: torch.Tensor) -> torch.Tensor:
        return -(x.unsqueeze(1) - x.unsqueeze(0)).abs().sum(dim=-1)

    @staticmethod
    def l2(x: torch.Tensor) -> torch.Tensor:
        return -(x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=-1).add(1e-8).sqrt()

    @staticmethod
    def dot(x: torch.Tensor) -> torch.Tensor:
        return x @ x.T

    @staticmethod
    def cos(x: torch.Tensor) -> torch.Tensor:
        normed = torch.nn.functional.normalize(x, p=2, dim=-1)
        return normed @ normed.T
    
    @staticmethod
    def tanh_dot(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ x.T)
    
    @staticmethod
    def atan_dot(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ x.T)
    
    @staticmethod
    def sqrt_dot(x: torch.Tensor) -> torch.Tensor:
        dot = x @ x.T
    
        return dot.sign() * dot.abs().sqrt()

SimilarityMetric.implementation_table = {
    "L1": SimilarityMetric.l1,
    "L2": SimilarityMetric.l2,
    "DOT": SimilarityMetric.dot,
    "COS": SimilarityMetric.cos,
    "ATAN_DOT": SimilarityMetric.atan_dot,
    "TANH_DOT": SimilarityMetric.tanh_dot,
    "SQRT_DOT": SimilarityMetric.tanh_dot
}