import torch
import torch.nn as nn


class SteeringWrapperModel(nn.Module):
    @classmethod
    def apply(cls, model: nn.Module):
        hidden_dim = model.config.hidden_size

        apply_model = model

        if hasattr(model, "model"):
            apply_model = model.model

        if hasattr(apply_model, "layers"):
            for layer in apply_model.layers:
                layer.inner_module = SteeringWrapperLayer(layer.inner_module, hidden_dim)
        else:
            raise ValueError("Model does not have layers")

        return cls(SteeringWrapperLayer(model, hidden_dim))

    def __init__(self, inner_module: nn.Module):
        super().__init__()
        self.inner_module = inner_module
    
    def forward(self, *args, **kwargs):
        return self.inner_module(*args, **kwargs)


class SteeringWrapperLayer(nn.Module):
    def __init__(self, inner_module: nn.Module, hidden_size: int):
        super().__init__()
        self.inner_module = inner_module

        self.hidden_size = hidden_size

        self.steering_vector = nn.Parameter(torch.randn(hidden_size))

    def forward(self, *kwargs):
        out = self.model(input_ids, attention_mask, **kwargs)

        return (out[0] + self.steering_vector, *out[1:])