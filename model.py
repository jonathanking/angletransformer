import torch
import torch.nn as nn
from openfold.model.primitives import Linear
from sidechainnet.examples.transformer import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AngleTransformer(nn.Module):
    """A shallow Transformer Encoder to predict torsion angles from AF2 seq embeding."""

    def __init__(
        self,
        c_s,
        c_hidden,
        no_blocks,
        no_angles,
        epsilon,
        dropout=0.1,
        d_ff=2048,
        no_heads=4,
        activation="relu",
    ):
        super().__init__()
        self.eps = epsilon
        self.c_hidden = c_hidden

        self.linear_initial = Linear(c_s, c_hidden)
        self.linear_in = Linear(c_s, c_hidden)
        self.relu = nn.ReLU()

        self.pos_encoder = PositionalEncoding(c_hidden)
        encoder_layers = TransformerEncoderLayer(
            c_hidden,
            nhead=no_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=no_blocks
        )

        self.linear_out = nn.Linear(c_hidden, no_angles * 2)

    def forward(self, s, s_initial):
        # [*, C_hidden], eg [1, 256 (L), 384  (c_s)]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        # Transformer in lieu of ResNet
        s = self.pos_encoder(s)
        s = self.transformer_encoder(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s  # Batch x Length x 7 x 2
