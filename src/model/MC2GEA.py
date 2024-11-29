import torch
from torch import nn, cosine_similarity
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings
import torch.nn.functional as F
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

    def recon_x_loss(self, x, reconstructed_x, embeddings, k = 256):
        """
        Calcule la perte de reconstruction et minimise l'écart des similarités cosinus entre X et Z.

        Args:
            x: Les vraies caractéristiques des nœuds.
            reconstructed_x: Les caractéristiques reconstruites des nœuds.
            embeddings: Les embeddings des nœuds.
            k: Nombre de paires les plus similaires à considérer.

        Returns:
            mse_loss: La perte de reconstruction.
            similarity_loss: La perte liée à l'écart de similarités.
        """
        # Perte MSE standard
        loss_fn = nn.MSELoss()
        mse_loss = loss_fn(reconstructed_x, x)

        # Calcul de la similarité cosinus dans X (caractéristiques originales)
        similarity_matrix_x = cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

        # Sélectionner les k paires les plus similaires dans X
        indices = torch.triu_indices(similarity_matrix_x.size(0), similarity_matrix_x.size(1), 1)
        values_x = similarity_matrix_x[indices[0], indices[1]]
        topk_values_x, topk_indices_x = torch.topk(values_x, k=min(k, len(values_x)), largest=True)

        # Embeddings des paires les plus similaires (dans Z)
        similar_pairs_x = [(indices[0][idx], indices[1][idx]) for idx in topk_indices_x]
        similarity_loss = 0
        for i, j in similar_pairs_x:
            # Similarité cosinus dans Z
            sim_z = cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), dim=-1)

            # Différence entre similarités dans X et Z
            similarity_loss += (sim_z - similarity_matrix_x[i, j]).abs()

        # Moyenne sur toutes les paires
        similarity_loss /= len(similar_pairs_x)

        # Retourner les pertes MSE et similarité
        return mse_loss, similarity_loss




    def contrastive_loss(self, H_1, H_2):
        c_l = ContrastiveLoss.ContrastiveLoss()
        cl_loss = c_l.contrastive_loss(H_1, H_2)
        return cl_loss

    def recon_r(self, e1,rel,e2):
        """

        :param e1: head entity
        :param rel: relation between e1 and e2
        :param e2: tail entity
        :return: score of triplet correctness
        """
        return self.r_decoder.forward(e1,rel,e2)

    def recon_r_loss(self, preds, labels):
        """
        :param preds: a tensor of prediction (triplet's score)
        :param labels: true labels
        :return: BCE loss
        """

        R_recons_loss =  F.binary_cross_entropy(preds, labels)
        return R_recons_loss