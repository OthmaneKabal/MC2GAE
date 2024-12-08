import torch
from torch import nn, cosine_similarity
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings
import torch.nn.functional as F
import loss_func


class MC2GEA(nn.Module):
    """
    Auto-encodeur de graphe pour la reconstruction des caractéristiques des nœuds.

    Args:
        encoder (nn.Module): Le module d'encodage.
        decoder (nn.Module): Le module de décodage.
    """

## options["X", "R", "cotrastive"]
    def __init__(self, encoder: nn.Module, x_decoder: nn.Module, r_decoder = None, projections = [500, 500]): #, options = ["X"]):
        super(MC2GEA, self).__init__()
        self.encoder = encoder
        self.x_decoder = x_decoder
        # self.reset_parameters()
        self.r_decoder = r_decoder
        # self.options = options
        self.projector_fc1 = nn.Sequential(nn.Linear(encoder.out_channels, projections[0], bias=True),
                                           nn.PReLU(),
                                           nn.Linear(projections[0], projections[1], bias=True)
                                           )
        self.projector_fc2 = nn.Sequential(nn.Linear(encoder.out_channels, projections[0], bias=True),
                                           nn.PReLU(),
                                           nn.Linear(projections[0], projections[1], bias=True)
                                           )


    def reset_parameters(self):
        """Réinitialise tous les paramètres apprenables du module."""
        self.encoder.reset_parameters()
        self.x_decoder.reset_parameters()


    def forward(self, data):
        return self.encode(data)


    def encode(self, data):
        embeddings = self.encoder(data)
        return embeddings


    def decode_x(self, data, embeddings):
        reconstructed_x = self.x_decoder(data, embeddings)
        return reconstructed_x


    def recon_r(self,e1,rel,e2):


        """

        :param e1: head entity
        :param rel: relation between e1 and e2
        :param e2: tail entity
        :return: score of triplet correctness
        """
        return self.r_decoder.forward(e1,rel,e2)