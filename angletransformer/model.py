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
        conv_encoder=False,
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

        if conv_encoder:
            self._setup_conv_encoder()

        self.linear_out = nn.Linear(c_hidden, no_angles * 2)
 
    def _setup_conv_encoder(self):
        # Create a 1D convolutional encoder to process the sequence, kernel size 11, that
        # preserves length and dimensionality
        self.conv_initial = nn.Conv1d(
            in_channels=self.c_hidden,
            out_channels=self.c_hidden,
            kernel_size=11,
            stride=1,
            padding=5,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.conv_current = nn.Conv1d(
            in_channels=self.c_hidden,
            out_channels=self.c_hidden,
            kernel_size=11,
            stride=1,
            padding=5,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.relu3 = nn.ReLU()

    def forward(self, s, s_initial):
        # [*, C_hidden], eg [1, 256 (L), 384  (c_s)]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)

        # Perform a 1D convolutional encoding of the sequence if requested
        if hasattr(self, "conv_initial"):
            # Conv for s_initial
            s_initial = s_initial.transpose(-1, -2)
            s_initial = self.conv_initial(s_initial)
            s_initial = s_initial.transpose(-1, -2)
            s_initial = self.relu2(s_initial)
            # Conv for s
            s = s.transpose(-1, -2)
            s = self.conv_current(s)
            s = s.transpose(-1, -2)
            s = self.relu3(s)

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
