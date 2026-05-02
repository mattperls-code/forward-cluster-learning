import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

class PredictionLayer(nn.Module):
    def __init__(
        self,
        input_feature_count: int,
        classification_count: int,
        seq_pooling_method: None | Literal["mean"] | Literal["last"] = None
    ):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_feature_count, classification_count))
        self.seq_pooling_method = seq_pooling_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_pooling_method is None:
            return F.softmax(self.encoder(x.flatten(start_dim=1)), dim=-1)
        
        elif self.seq_pooling_method == "mean":
            return F.softmax(self.encoder(x.mean(dim=1)), dim=-1)
        
        else:
            return F.softmax(self.encoder(x[:, -1, :]), dim=-1)

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

    def to(self, device: torch.device):
        self.model.to(device)

        for optimizer in self.optimizers:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
                        
        return self

    def reset(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for optimizer in self.optimizers:
            optimizer.state.clear()
            
    # x (batch size, input size) -> prediction index (batch size, ), global probabilities (batch size, num classes), per layer probabilities (batch size, num layers, num classes)
    def forward(self, x: torch.Tensor, output_intermediate_predictions=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()

        segment_predictions = []

        with torch.no_grad():
            for prediction_segment in self.prediction_segments:
                for layer in prediction_segment:
                    if isinstance(layer, PredictionLayer): segment_predictions.append(layer.predict(x))

                    x = layer(x)

        # ensemble_prediction = torch.stack(segment_predictions, dim=1).sum(dim=1).argmax(dim=-1)

        return segment_predictions if output_intermediate_predictions else segment_predictions[-1].argmax(dim=-1)

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