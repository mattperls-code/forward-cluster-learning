import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PredictionLayer(nn.Module):
    def __init__(
        self,
        input_feature_count: int,
        classification_count: int,
        trustworthiness: float = 1.0,
    ):
        super().__init__()

        self.trustworthiness = nn.Parameter(torch.tensor(trustworthiness))

        self.encoder = nn.Sequential(
            nn.Linear(input_feature_count, input_feature_count // 2),
            nn.ReLU(),
            nn.Linear(input_feature_count // 2, classification_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.trustworthiness * F.softmax(self.encoder(x), dim=-1)

class ForwardClusterLearning:
    def __init__(
        self,
        model: nn.Sequential,
        optimizer_class: type,
        optimizer_kwargs: Optional[dict] = None,
    ):
        optimizer_kwargs = optimizer_kwargs or {}
        layers = list(model.children())

        if not isinstance(layers[-1], PredictionLayer): raise ValueError( "The last layer of the model must be a PredictionLayer.")

        self.model = model

        # partition into prediction_segments, each ending with a PredictionLayer
        self.prediction_segments: list[list[nn.Module]] = []

        current_prediction_segment: list[nn.Module] = []
        for layer in layers:
            current_prediction_segment.append(layer)

            if isinstance(layer, PredictionLayer):
                self.prediction_segments.append(current_prediction_segment)

                current_prediction_segment = []

        self.optimizers: list[torch.optim.Optimizer] = []

        for i, prediction_segment in enumerate(self.prediction_segments):
            segment_params = [ params for layer in prediction_segment for params in layer.parameters() ]

            if not segment_params: raise ValueError(f"Prediction Segment {i} has no trainable parameters.")

            self.optimizers.append(optimizer_class(segment_params, **optimizer_kwargs))
            
    # x (batch size, input size) -> prediction index (batch size, ), global probabilities (batch size, num classes), per layer probabilities (batch size, num layers, num classes)
    def forward(self, x: torch.Tensor, output_intermediate_predictions=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()

        per_layer_probs = []

        with torch.no_grad():
            for prediction_segment in self.prediction_segments:
                for layer in prediction_segment:
                    if isinstance(layer, PredictionLayer): per_layer_probs.append(layer.predict(x))

                    x = layer(x)

        per_layer_probs = torch.stack(per_layer_probs, dim=1)
        global_probs = per_layer_probs.sum(dim=1)
        prediction_index = global_probs.argmax(dim=-1)

        return (prediction_index, global_probs, per_layer_probs) if output_intermediate_predictions else prediction_index

    # x (batch size, input size), expected_output (batch size, ) -> total_cce_loss (1, )
    def backward(self, x: torch.Tensor, expected_output: torch.Tensor) -> torch.Tensor:
        self.model.train()

        total_loss = torch.tensor(0.0, device=x.device)

        x = x.detach()

        for prediction_segment, optimizer in zip(self.prediction_segments, self.optimizers):
            segment_probabilities = None

            for layer in prediction_segment:
                if isinstance(layer, PredictionLayer): segment_probabilities = layer.predict(x)

                x = layer(x)

            loss = F.cross_entropy(segment_probabilities, expected_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.detach()

            x = x.detach()

        return total_loss