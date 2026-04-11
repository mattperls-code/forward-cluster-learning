import torch
from similarity_metrics import SimilarityMetric

"""
        Training occurs pairwise, for every training point in consideration.
        Fully contrastive, per layer loss.

        Moderate success, consistently a bit worse than bp.
        Likely fails to compose nonlinear features.
        Most of the work occurs in the classification head.
"""

class ForwardClusterLearningV1:
    def __init__(
        self,

        # underlying model to be trained
        model: torch.nn.Module,

        # optimizer to handle per-layer training
        optimizer: torch.optim.Optimizer,

        # parameters to pass to each layer's isolated optimizer
        optimizer_kwargs: dict | None,

        # loss function used to train final classification head
        loss_function: callable,

        # choice of per-layer similarity definition
        similarity_metric: SimilarityMetric,

        # number of initial layers to be trained with backpropagation, using loss derived by the following layer's similarity function
        backpropagation_cutoff: int = 0
    ):
        self.loss_function = loss_function
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
        labels: torch.Tensor,

        debug_logs=False
    ):
        unique_pairs_mask = torch.triu(torch.ones(inputs.shape[0], inputs.shape[0], dtype=torch.bool), diagonal=1)
        same_classification_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        current_layer_state = inputs

        # TODO: should handle non-parameterized final layer more efficiently by clipping earlier
        for _, layer in list(self.layers.items())[:-1]:
            if debug_logs: print(current_layer_state)

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

        similarity_matrix = self.similarity_metric(current_layer_state)
                
        loss = (similarity_matrix[~same_classification_mask & unique_pairs_mask].sum() - similarity_matrix[same_classification_mask & unique_pairs_mask].sum()) / unique_pairs_mask.sum()

        return loss

    # rebuild the final classification head to reflect the provided training data, treating the final layer as an independent regression task on the subsequent layer's output
    def build_classification_head(
        self,
        
        # (batch_size, input_dim)
        inputs: torch.Tensor,

        # (batch_size,)
        labels: torch.Tensor,

        # number of times to step the optimizer
        optimizer_iterations
    ):
        output_layer_name, output_layer = next(
            (layer_name, layer) for layer_name, layer in reversed(self.layers.items())
            if layer["optimizer"] is not None
        )

        output_module = output_layer["module"]
        output_optimizer = output_layer["optimizer"]

        fixed_point_layer_state = inputs

        with torch.no_grad():
            for layer_name, layer in self.layers.items():
                if layer_name == output_layer_name: break

                fixed_point_layer_state = layer["module"](fixed_point_layer_state)

        loss = None

        for _ in range(optimizer_iterations):
            current_layer_state = output_module(fixed_point_layer_state)

            reached = False

            for layer_name, layer in self.layers.items():
                if reached:
                    current_layer_state = layer["module"](current_layer_state)

                if layer_name == output_layer_name:
                    reached = True

            loss = self.loss_function(current_layer_state, labels)

            output_optimizer.zero_grad()
            loss.backward()
            output_optimizer.step()

        return loss.detach()