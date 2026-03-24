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

class ForwardClusterLearning:
    def __init__(
        self,

        # underlying model to be trained
        model: torch.nn.Module,

        # optimizer to handle per-layer training
        optimizer: torch.optim.Optimizer,

        # parameters to pass to each layer's isolated optimizer
        optimizer_kwargs: dict | None,

        # choice of per-layer similarity definition
        similarity_metric: SimilarityMetric,

        # number of initial layers to be trained with backpropagation, using loss derived by the following layer's similarity function
        backpropagation_cutoff: int = 0
    ):
        self.similarity_metric = similarity_metric
        self.backpropagation_cutoff = backpropagation_cutoff

        self.layers = {
            layer_name: {
                "module": module,
                "optimizer": optimizer(module.parameters(), **optimizer_kwargs) if list(module.parameters()) else None
            }
            for layer_name, module in model.named_children()
        }

    # perform a single cluster training iteration, which should adjust model parameters to better cluster/separate the given training batch
    def cluster(
        self,

        # (batch_size, input_dim)
        inputs: torch.Tensor,

        # (batch_size,)
        labels: torch.Tensor
    ):
        unique_pairs_mask = torch.triu(torch.ones(inputs.shape[0], inputs.shape[0], dtype=torch.bool), diagonal=1)
        same_classification_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        current_layer_state = inputs

        for _, layer in self.layers.items():
            module = layer["module"]
            optimizer = layer["optimizer"]

            current_layer_state = module(current_layer_state)

            if optimizer is not None:
                similarity_matrix = self.similarity_metric(current_layer_state)
                
                loss = (similarity_matrix[~same_classification_mask & unique_pairs_mask].sum() - similarity_matrix[same_classification_mask & unique_pairs_mask].sum()) / unique_pairs_mask.sum()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            current_layer_state = current_layer_state.detach()

        print(current_layer_state)
        print()

    # rebuild the final classification head to reflect the provided training data, treating the final layer as an independent regression task on the subsequent layer's output
    def build_classification_head(
        self,

        # list of labeled training examples, each containing an input tensor and correct classification label
        training_data: list[tuple[torch.Tensor, int]]
    ):
        pass