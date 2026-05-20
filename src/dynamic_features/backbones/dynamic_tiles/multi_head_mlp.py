import torch
from torch import Tensor, nn


class MultiHeadMLP(nn.Module):
    """MLP with a shared backbone feeding into multiple independent linear heads.

    The backbone is a stack of ``Linear → ReLU → Dropout`` layers. Each head
    is a single linear layer that projects the backbone output to
    ``n_output_per_head`` values. All heads run in parallel and their outputs
    are stacked into a single tensor.

    Useful when the same feature extractor should produce predictions for
    several related tasks simultaneously (e.g., one cluster-classification
    head per tiling in ``ClusteringFeatures``).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden_layers_backbone: int,
        hidden_dim: int,
        n_heads: int,
        n_output_per_head: int,
        dropout: float = 0.0,
    ) -> None:
        """Builds a multi-head MLP with a shared backbone and independent output heads.

        Args:
            n_input: Number of input features.
            n_hidden_layers_backbone: Number of hidden linear layers in the shared
                backbone. If 0, the input is passed directly to each head.
            hidden_dim: Width of each backbone hidden layer.
            n_heads: Number of independent output heads.
            n_output_per_head: Number of output features produced by each head.
            dropout: Dropout probability applied after each backbone activation.
                0 disables dropout.
        """
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.backbone = nn.ModuleList()
        in_features = n_input
        for _ in range(n_hidden_layers_backbone):
            self.backbone.append(nn.Linear(in_features=in_features, out_features=hidden_dim))
            in_features = hidden_dim

        self.heads = nn.ModuleList(
            [nn.Linear(in_features=in_features, out_features=n_output_per_head) for _ in range(n_heads)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs a forward pass through the shared backbone then all heads.

        Each backbone layer applies ``Linear → ReLU → Dropout`` in sequence.
        The resulting representation is then passed to every head independently,
        and the per-head outputs are stacked along the second-to-last dimension.

        Args:
            x: Input tensor of shape (batch, n_input) or (n_input,) for a
                single sample.

        Returns:
            Tensor of shape (batch, n_heads, n_output_per_head) for batch input,
            or (n_heads, n_output_per_head) for a single sample.
        """
        for layer in self.backbone:
            x = self.activation(layer(x))
            x = self.dropout(x)

        out = [head(x) for head in self.heads]
        return torch.stack(out, dim=-2)

    def freeze_backbone(self) -> None:
        """Freezes all backbone parameters by setting ``requires_grad=False``.

        Subsequent optimizer steps will not update the backbone. Does not
        affect the forward pass or inference.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Restores backbone parameters to trainable by setting ``requires_grad=True``."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def reset_heads(self) -> None:
        """Re-initializes all head weights and biases to their default values.

        Each head is a single ``nn.Linear`` layer; this calls
        ``reset_parameters()`` on every head, which re-applies Kaiming uniform
        initialization. Useful before retraining heads on a new set of cluster
        assignments.
        """
        for head in self.heads:
            head.reset_parameters()
