import torch
from torch import nn, cosine_similarity
import torch.nn.functional as F
import loss_func


class MRGAE(nn.Module):
    """
    Auto-encodeur de graphe pour la reconstruction des caractéristiques des nœuds.

    Args:
        encoder (nn.Module): Le module d'encodage.
        decoder (nn.Module): Le module de décodage.
    """

## options["X", "R", "cotrastive"]
    def __init__(self, encoder: nn.Module, x_decoder: nn.Module, r_decoder = None, projections = None): #, options = ["X"]):
        super(MRGAE, self).__init__()
        self.encoder = encoder
        self.x_decoder = x_decoder
        # self.reset_parameters()
        self.r_decoder = r_decoder
        # self.options = options
        ##  projections for contrastive
        if projections:
            print("hhh")
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


    def forward(self, data, return_projected=False):
        """
        Si return_projected=True, retourne aussi les projections contrastives.
        """
        embeddings = self.encode(data)
        if return_projected and self.use_projection:
            proj1 = self.projector_fc1(embeddings)
            proj2 = self.projector_fc2(embeddings)
            return embeddings, proj1, proj2
        return embeddings


    def encode(self, data):
        embeddings = self.encoder(data)
        return embeddings


    def decode_x(self, data, embeddings, r_embeddings = None):
        if r_embeddings is None:
            reconstructed_x = self.x_decoder(data, embeddings)
        else:
            reconstructed_x = self.x_decoder(data, embeddings, r_embeddings)

        return reconstructed_x


    def recon_r(self,e1,rel,e2):
        """

        :param e1: head entity
        :param rel: relation between e1 and e2
        :param e2: tail entity
        :return: score of triplet correctness
        """
        return self.r_decoder.forward(e1,rel,e2)


    def recon_r_(self,z, edge_index, edge_type):
        """

        :param e1: head entity
        :param rel: relation between e1 and e2
        :param e2: tail entity
        :return: score of triplet correctness
        """
        return self.r_decoder.forward(z, edge_index, edge_type)