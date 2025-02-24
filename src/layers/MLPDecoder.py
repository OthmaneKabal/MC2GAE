import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, data, alpha=0.01, dropout=0.5):
        """
        Initialize the MLP-based decoder with the same structure as the GCNDecoder.

        Parameters:
        - encoder: The encoder used to obtain embeddings,
                   from which we extract the layer dimensions.
        - data: Data object to access the original features (data.x).
        - alpha: Leaky ReLU coefficient to preserve negative values (not used in MLP but for consistency).
        - dropout: Dropout probability for regularization.
        """
        super(MLPDecoder, self).__init__()

        # Get dimensions of the encoder layers to reverse them for the decoder
        # encoder_out_channels = [layer.out_features for layer in encoder.fcs]
        encoder_out_channels = [layer.out_channels for layer in encoder.convs]

        encoder_in_channels = data.num_features # Initial node feature dimension

        # Reverse encoder dimensions for the decoder
        decoder_out_channels = list(reversed(encoder_out_channels)) + [encoder_in_channels]

        # Create fully connected layers for the decoder
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch normalization layers
        for i in range(len(decoder_out_channels) - 1):
            input_dim = decoder_out_channels[i]
            output_dim = decoder_out_channels[i + 1]
            self.fcs.append(nn.Linear(input_dim, output_dim))
            self.bns.append(nn.BatchNorm1d(output_dim))

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Final linear layer to reconstruct node features
        self.final_layer = nn.Linear(decoder_out_channels[-1], data.num_features)

        # Activation function
        self.relu = nn.ReLU()

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
    def reset_parameters(self):
        """Reset the parameters of the decoder layers."""
        for fc in self.fcs:
            fc.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, data,embeddings):
        """
        Forward pass in the decoder to reconstruct node features.

        Parameters:
        - embeddings: Embeddings produced by the encoder (input to the decoder).

        Returns:
        - Reconstructed node features.
        """
        x = embeddings

        # Apply each MLP layer with BatchNorm, ReLU, and Dropout in between
        for fc, bn in zip(self.fcs, self.bns):
            x = fc(x)
            x = bn(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        # Apply the final linear layer
        x = self.final_layer(x)
        return x

