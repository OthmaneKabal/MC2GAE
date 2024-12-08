import torch
import torch.nn.functional as F
from torch import nn
from torch import nn, cosine_similarity


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.5):
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        # Normalisation des embeddings
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        # Calcul de la similarité cosinus
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        # Fonction pour appliquer l'exponentielle avec la température
        f = lambda x: torch.exp(x / self.tau)

        # Similarité cosinus entre les embeddings dans la même vue (référence)
        refl_sim = f(self.sim(z1, z1))

        # Similarité cosinus entre les embeddings des vues différentes
        between_sim = f(self.sim(z1, z2))

        # Calcul de la perte InfoNCE
        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        ).mean()

    def contrastive_loss(self, H_1: torch.Tensor, H_2: torch.Tensor):
        # Calcul de la perte semi-contrastive entre H_1 et H_2
        l1 = self.semi_loss(H_1, H_2)
        l2 = self.semi_loss(H_2, H_1)
        # Moyenne des deux pertes pour la symétrie
        return (l1 + l2) / 2

def similarity_pair_loss(x, reconstructed_x, embeddings, k=None):
    """
    Calcule la perte de reconstruction et minimise l'écart des similarités cosinus entre X et Z.

    Args:
        x: Les vraies caractéristiques des nœuds.
        reconstructed_x: Les caractéristiques reconstruites des nœuds.
        embeddings: Les embeddings des nœuds.
        k: Nombre de paires les plus similaires à considérer.

    Returns:
        similarity_loss: La perte liée à l'écart de similarités.
    """
    # Calcul de la similarité cosinus dans X (caractéristiques originales)
    similarity_matrix_x = cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
    # Calcul de la similarité cosinus dans les embeddings
    similarity_matrix_z = cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    # Sélectionner les k paires les plus similaires dans X (au-dessus de la diagonale uniquement)
    indices = torch.triu_indices(similarity_matrix_x.size(0), similarity_matrix_x.size(1), 1)
    values_x = similarity_matrix_x[indices[0], indices[1]]
    # Si k est inférieur au nombre de paires, sélectionne les k plus grandes similarités
    if k:
        topk_values_x, topk_indices = torch.topk(values_x, k=k, largest=True)
        selected_indices = (indices[0][topk_indices], indices[1][topk_indices])
    else:
        selected_indices = indices
    # Récupérer les similarités cosinus correspondantes dans Z
    sim_x = similarity_matrix_x[selected_indices[0], selected_indices[1]]
    sim_z = similarity_matrix_z[selected_indices[0], selected_indices[1]]
    # Calcul de la perte : différence absolue entre similarités dans X et Z
    similarity_loss = torch.mean(torch.abs(sim_z - sim_x))
    # Retourner les pertes MSE et similarité
    return similarity_loss

def sce_loss_fnc(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def contrastive_loss(H_1, H_2):
    c_l = ContrastiveLoss()
    cl_loss = c_l.contrastive_loss(H_1, H_2)
    return cl_loss

def mse_loss_fnc(x, reconstructed_x):
    # Perte MSE standard
    loss_fn = nn.MSELoss()
    mse_loss = loss_fn(reconstructed_x, x)
    return mse_loss


def recon_r_loss(preds, labels):
    """
    :param preds: a tensor of prediction (triplet's score)
    :param labels: true labels
    :return: BCE loss
    """

    R_recons_loss =  F.binary_cross_entropy(preds, labels)
    return R_recons_loss
