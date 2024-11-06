import torch
from torch_geometric.data import Data

def view_partial_features_masking(data, max_masking_percentage=0.3):
    """
    Génère une vue augmentée du graphe en appliquant un masquage partiel exact
    sur les caractéristiques des nœuds, respectant le pourcentage de masquage pour chaque nœud.

    Paramètres:
    - data : objet de type Data avec les attributs x, edge_index, edge_attr, et edge_type
    - max_masking_percentage : pourcentage maximal de masquage pour chaque nœud (par défaut 20%)

    Retourne:
    - data_augmented : une copie de l'objet Data avec les caractéristiques des nœuds masquées
    """
    # Vérifier l'appareil et déplacer tous les éléments de data sur le même appareil
    device = data.x.device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(device)
    if data.edge_type is not None:
        data.edge_type = data.edge_type.to(device)

    # Récupérer le nombre de nœuds et de caractéristiques
    num_nodes, num_features = data.x.shape

    # Générer un taux de masquage aléatoire pour chaque nœud entre 0 et max_masking_percentage
    node_masking_ratios = (torch.rand(num_nodes, device=device) * max_masking_percentage)

    # Créer un masque exact pour chaque nœud
    mask = torch.ones((num_nodes, num_features), device=device)
    for i in range(num_nodes):
        # Calculer le nombre exact de caractéristiques à masquer pour le nœud i
        num_features_to_mask = int(num_features * node_masking_ratios[i].item())

        # Choisir aléatoirement les indices des caractéristiques à masquer
        mask_indices = torch.randperm(num_features, device=device)[:num_features_to_mask]

        # Appliquer le masquage aux indices sélectionnés
        mask[i, mask_indices] = 0

    # Appliquer le masque aux caractéristiques pour obtenir la vue partiellement masquée
    masked_x = data.x * mask

    # Créer une copie de data pour la vue augmentée avec les caractéristiques masquées
    data_augmented = data.clone()
    data_augmented.x = masked_x

    return data_augmented

def relation_based_edge_dropping(data, total_drop_rate):
    # Créer des copies pour éviter de modifier l'objet data original
    edge_index = data.edge_index.clone()
    edge_type = data.edge_type.clone()

    # Calcul de la fréquence de chaque type de relation
    edge_types, edge_counts = torch.unique(edge_type, return_counts=True)
    total_edges = edge_index.size(1)

    # Calcul du nombre total d'arêtes à supprimer
    num_edges_to_drop = int(total_edges * total_drop_rate)

    # Calcul du nombre d'arêtes à supprimer par type de relation
    edges_to_drop_per_type = {edge_type.item(): int(num_edges_to_drop * (count.item() / total_edges))
                              for edge_type, count in zip(edge_types, edge_counts)}

    # Liste pour stocker les indices des arêtes à garder
    keep_edge_indices = list(range(total_edges))

    # Pour chaque type de relation, tirer aléatoirement les indices d'arêtes à supprimer
    for edge_type_value, edges_to_drop in edges_to_drop_per_type.items():
        # Trouver les indices des arêtes de ce type
        edge_indices_of_type = torch.where(edge_type == edge_type_value)[0].tolist()

        # Tirer aléatoirement le nombre requis d'indices à supprimer
        indices_to_remove = []
        while len(indices_to_remove) < edges_to_drop and edge_indices_of_type:
            candidate = edge_indices_of_type.pop(torch.randint(0, len(edge_indices_of_type), (1,)).item())

            # Vérifier la contrainte de non-isolation des nœuds
            src, dst = edge_index[:, candidate]
            if torch.sum(edge_index == src).item() > 1 and torch.sum(edge_index == dst).item() > 1:
                indices_to_remove.append(candidate)

        # Supprimer les indices sélectionnés du jeu d'arêtes à garder
        keep_edge_indices = [idx for idx in keep_edge_indices if idx not in indices_to_remove]

    # Création d'un nouvel objet Data avec les arêtes mises à jour
    new_data = Data(
        x=data.x,  # Copie des noeuds
        edge_index=edge_index[:, keep_edge_indices],  # Arêtes mises à jour
        edge_type=edge_type[keep_edge_indices],  # Types d'arêtes mis à jour
    )

    return new_data
