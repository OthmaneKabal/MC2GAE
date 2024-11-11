import torch
from torch import nn
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings

import ContrastiveLoss



class MC2GEA(nn.Module):
    """
    Auto-encodeur de graphe pour la reconstruction des caractéristiques des nœuds.

    Args:
        encoder (nn.Module): Le module d'encodage.
        decoder (nn.Module): Le module de décodage.
    """


    def __init__(self, encoder: nn.Module, x_decoder: nn.Module, r_decoder = None, contrastive = False):
        super(MC2GEA, self).__init__()
        self.encoder = encoder
        self.x_decoder = x_decoder
        self.reset_parameters()
        self.contrastive = contrastive


    def reset_parameters(self):
        """Réinitialise tous les paramètres apprenables du module."""
        self.encoder.reset_parameters()
        self.x_decoder.reset_parameters()


    def forward(self, data):
        return self.encode(data)


    def encode(self, data):
        if self.contrastive:
            embeddings = {}
            H_1 = self.encoder(data["G1"])
            H_2 = self.encoder(data["G2"])
            embeddings["H1"] = H_1
            embeddings["H2"] = H_2

            return embeddings
        else:
            embeddings = self.encoder(data)
            return embeddings


    def decode_x(self, data, embeddings):
        reconstructed_x = self.x_decoder(data, embeddings)
        return reconstructed_x


    def recon_x_loss(self, data, reconstructed_x):
        loss_fn = nn.MSELoss()
        return loss_fn(reconstructed_x, data.x)


    def contrastive_loss(self, embeddings):
        c_l = ContrastiveLoss.ContrastiveLoss()
        c_l.contrastive_loss(embeddings["H1"], embeddings["H2"])
        return c_l

