import torch
from torch import nn, cosine_similarity
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings

import ContrastiveLoss



class MC2GEA(nn.Module):
    """
    Auto-encodeur de graphe pour la reconstruction des caractéristiques des nœuds.

    Args:
        encoder (nn.Module): Le module d'encodage.
        decoder (nn.Module): Le module de décodage.
    """

## options["X", "R", "cotrastive"]
    def __init__(self, encoder: nn.Module, x_decoder: nn.Module, r_decoder = None, projections = [256, 256]): #, options = ["X"]):
        super(MC2GEA, self).__init__()
        self.encoder = encoder
        self.x_decoder = x_decoder
        # self.reset_parameters()
        self.r_decoder = r_decoder
        # self.options = options
        # self.projector_fc1 = nn.Sequential(nn.Linear(encoder.out_channels, projections[0], bias=True),
        #                                    nn.PReLU(),
        #                                    nn.Linear(projections[0], projections[1], bias=True)
        #                                    )
        # self.projector_fc2 = nn.Sequential(nn.Linear(encoder.out_channels, projections[0], bias=True),
        #                                    nn.PReLU(),
        #                                    nn.Linear(projections[0], projections[1], bias=True)
        #                                    )

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


    def recon_x_loss(self, x, reconstructed_x, embeddings, k = 15):
        # loss_fn = nn.MSELoss()
        # return loss_fn(reconstructed_x, x)

        loss_fn = nn.MSELoss()
        mse_loss = loss_fn(reconstructed_x, x)

        # Calcul de la similarité cosinus entre tous les nœuds du batch
        similarity_matrix = cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

        # Sélectionner les k paires de nœuds les plus similaires
        indices = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), 1)
        values = similarity_matrix[indices[0], indices[1]]
        topk_values, topk_indices = torch.topk(values, k=min(k, len(values)), largest=True)

        # Embeddings des nœuds les plus similaires
        similar_pairs = [(indices[0][idx], indices[1][idx]) for idx in topk_indices]
        cos_loss = 0
        for i, j in similar_pairs:
            cos_loss += 1 - cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), dim=-1).mean()

        # Moyenne sur toutes les paires
        cos_loss /= len(similar_pairs)

        # Combiner la perte MSE et la perte cosinus
        # total_loss = mse_loss + config["cosine_loss_weight"] * cos_loss
        return mse_loss, cos_loss




    def contrastive_loss(self, H_1, H_2):
        c_l = ContrastiveLoss.ContrastiveLoss()
        cl_loss = c_l.contrastive_loss(H_1, H_2)
        return cl_loss

    def recon_r(self, ):
        pass


    def recon_r_loss(self):
        pass
