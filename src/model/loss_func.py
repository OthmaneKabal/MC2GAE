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

    def semi_loss_exclude(self, z1: torch.Tensor, z2: torch.Tensor, edge_index: torch.Tensor, excluded: torch.Tensor):
        """
        Calcul de la perte semi-contrastive en excluant les paires définies par un masque, sans boucle.

        Args:
            z1: Embeddings de la première vue.
            z2: Embeddings de la seconde vue.
            edge_index: Tenseur (2, nb_links) contenant les indices des arêtes.
            excluded: Masque binaire (nb_links), 1 pour exclure une arête, 0 sinon.

        Returns:
            torch.Tensor: Perte semi-contrastive.
        """
        # Fonction exponentielle avec la température
        f = lambda x: torch.exp(x / self.tau)

        # Similarités cosinus intra-vue et inter-vue
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        # Construction du masque global
        num_nodes = z1.size(0)
        mask = torch.ones((num_nodes, num_nodes), device=z1.device)  # Initialise un masque complet

        # Masque binaire pour les indices dans edge_index
        excluded_indices = edge_index[:, excluded.bool()]  # Sélectionne les arêtes à exclure
        mask[excluded_indices[0], excluded_indices[1]] = 0  # Exclut les paires (source, target)
        mask[excluded_indices[1], excluded_indices[0]] = 0  # Symétrie pour un graphe non orienté
        print(mask.shape)
        # Application du masque sur les similarités
        refl_sim = refl_sim * mask
        between_sim = between_sim * mask

        # Calcul de la perte InfoNCE avec le dénominateur masqué
        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        ).mean()

    def contrastive_loss_exclude(self, H_1: torch.Tensor, H_2: torch.Tensor, edge_index: torch.Tensor, excluded: torch.Tensor):
        l1 = self.semi_loss_exclude(H_1, H_2, edge_index, excluded)
        l2 = self.semi_loss_exclude(H_2, H_1, edge_index, excluded)
        return l1 + l2 / 2
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


def contrastive_loss_exclude_is(H_1, H_2, edge_index, excluded):
    c_l = ContrastiveLoss()
    cl_loss = c_l.contrastive_loss_exclude(H_1, H_2, edge_index, excluded)
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


import torch
import torch.nn.functional as F


def calculate_cluster_assignments(node_embeddings, core_concepts):
    """
    Calcule les assignations de clusters pour chaque nœud en fonction de la similarité cosinus.

    Arguments:
    - node_embeddings (torch.Tensor): Les embeddings des nœuds, de dimension [N, d], où N est le nombre de nœuds et d est la dimension des embeddings.
    - core_concepts (torch.Tensor): Les embeddings des concepts centraux des clusters, de dimension [num_clusters, d].

    Retourne:
    - cluster_assignments (torch.Tensor): Les indices des clusters associés aux nœuds, de dimension [N].
    """
    # Calculer la similarité cosinus entre chaque nœud et tous les concepts centraux
    # Résultat : [N, num_clusters]
    cosine_similarities = F.cosine_similarity(
        node_embeddings.unsqueeze(1),  # Ajout d'une dimension pour comparer avec chaque core concept
        core_concepts.unsqueeze(0),  # Alignement des dimensions pour le calcul
        dim=-1
    )

    # Trouver l'indice du cluster ayant la similarité maximale pour chaque nœud
    cluster_assignments = cosine_similarities.argmax(dim=1)
    max_similarities = cosine_similarities.max(dim=1).values

    return cluster_assignments, max_similarities
    # return cluster_assignments


def inter_cluster_loss(node_embeddings, cluster_assignments, core_concepts):

    cluster_core_embeddings = core_concepts[cluster_assignments]
    cosine_similarities = F.cosine_similarity(node_embeddings, cluster_core_embeddings, dim=-1)
    losses = 1 - cosine_similarities
    inter_cluster_loss = losses.mean()
    if (inter_cluster_loss<0):
        print("********interloss*****")
    return inter_cluster_loss

def intra_cluster_loss(core_concepts_embeddings, scale_factor=1.0):
    """
    Calcule la perte intra-cluster pour renforcer la séparation entre les clusters.

    Arguments:
    - core_concepts (torch.Tensor): Les embeddings des concepts centraux, de dimension [num_clusters, d].
    - scale_factor (float): Un facteur d'échelle (par défaut 1.0).

    Retourne:
    - intra_cluster_loss (torch.Tensor): La perte intra-cluster moyenne.
    """
    num_clusters = core_concepts_embeddings.size(0)  # Nombre de core concepts

    if num_clusters < 2:
        # Aucun calcul à effectuer si moins de 2 clusters
        return torch.tensor(0.0, device=core_concepts_embeddings.device)

    # Calculer toutes les similarités cosinus entre les core concepts
    cosine_similarities = F.cosine_similarity(
        core_concepts_embeddings.unsqueeze(1),  # [num_clusters, 1, d]
        core_concepts_embeddings.unsqueeze(0),  # [1, num_clusters, d]
        dim=-1
    )  # Résultat : [num_clusters, num_clusters]

    # Masquer les diagonales (similarités entre le même core concept)
    mask = torch.eye(num_clusters, device=core_concepts_embeddings.device).bool()
    cosine_similarities = cosine_similarities.masked_fill(mask, 0)
    # Somme des similarités cosinus entre les différents core concepts
    total_similarity = cosine_similarities.sum()

    # Calcul de la perte normalisée
    loss = scale_factor * total_similarity / (num_clusters * (num_clusters - 1))

    return loss
