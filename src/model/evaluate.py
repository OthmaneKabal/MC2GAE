from data.GraphDataPreparation import GraphDataPreparation
from src.layers.RGCNDecoder import RGCNDecoder
from src.layers.RGCNEncoder import RGCNEncoder
from src.model.MC2GEA import MC2GEA
from src.model.utils import save_model, load_model_checkpoint
from config import config
import torch
from torch import optim
from torch_geometric.data import Data
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def generate_gs_embeddings(graph_path, checkpoint_path, gs_path, core_concepts, config,embedding_model = "GNN"):
    """
    Charge le graphe, initialise et charge le modèle à partir du checkpoint, et génère les embeddings
    pour les termes présents dans le GS ainsi que pour les concepts principaux spécifiés.

    :param graph_path: Chemin vers le fichier JSON du graphe
    :param checkpoint_path: Chemin vers le checkpoint du modèle
    :param gs_path: Chemin vers le fichier Excel contenant le GS
    :param core_concepts: Liste des concepts principaux pour lesquels obtenir les embeddings
    :param config: Dictionnaire de configuration pour le modèle et l'optimiseur
    :return: Deux dictionnaires - un pour les termes du GS et un pour les core concepts,
             avec les termes/concepts comme clés et leurs embeddings comme valeurs
    """

    # Charger le graphe avec GraphDataPreparation
    gdp = GraphDataPreparation(config["Entities_path"], graph_path,
                               edges_embd_path=config["Edges_path"], is_directed=True)
    data = gdp.prepare_graph_with_type()
    data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(config["device"])
    print(data)

    if embedding_model == "GNN":
    # Initialisation du modèle
        RGCN_encoder = RGCNEncoder(data, config["out_channels"], config["num_layers"], config["num_bases"]).to(
            config["device"])
        RGCN_decoder = RGCNDecoder(RGCN_encoder, data, config["num_bases"], config["alpha"]).to(config["device"])
        autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder).to(config["device"])

        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

        # Charger le modèle et l'optimiseur à partir du checkpoint
        model, optimizer, start_epoch = load_model_checkpoint(autoencoder, optimizer, checkpoint_path)

        # Mettre le modèle en mode évaluation
        model.eval()

        # Encoder le graphe
        with torch.no_grad():
            embeddings = model.encode(data)

    else:
        embeddings = data.x


    # Charger le fichier GS
    gs_df = pd.read_excel(gs_path, sheet_name='Sheet1')
    gs_terms = set(gs_df['term'].str.lower().unique())  # Termes du GS en minuscules pour uniformiser

    # Décoder les indices des nœuds pour obtenir le texte associé
    node_index_to_text = gdp.decode_indexes()

    # Créer deux dictionnaires pour stocker les embeddings des termes présents dans le GS et des core concepts
    gs_embeddings = {}
    core_concepts_embeddings = {}
    # print(node_index_to_text)
    # Boucle pour récupérer les embeddings des termes présents dans le GS
    for node_id, term in node_index_to_text.items():
        term_lower = term.lower()
        # Vérifier si le terme est dans le GS
        if term_lower in gs_terms:
            gs_embeddings[term] = embeddings[node_id].cpu().numpy()  # Extraire l'embedding et le stocker

        # Vérifier si le terme est dans les core concepts
        if term_lower in [concept.lower() for concept in core_concepts]:
            core_concepts_embeddings[term] = embeddings[node_id].cpu().numpy()
    #
    # print(f"Nombre de termes du GS avec embeddings : {len(gs_embeddings)}")
    # print(f"Nombre de core concepts avec embeddings : {len(core_concepts_embeddings)}")

    return gs_embeddings, core_concepts_embeddings


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def classify_terms_by_cosine_similarity(gs_embeddings, core_concepts_embeddings, threshold=0.5):
    """
    Classe les termes du GS en fonction de leur similarité cosinus avec les core concepts.

    :param gs_embeddings: Dictionnaire avec les termes du GS comme clés et leurs embeddings comme valeurs
    :param core_concepts_embeddings: Dictionnaire avec les core concepts comme clés et leurs embeddings comme valeurs
    :param threshold: Seuil de similarité cosinus pour la classification
    :return: Dictionnaire avec chaque terme du GS, son core concept le plus proche, et la classe prédite
    """
    similarity = []
    classifications = {}

    # Boucle sur chaque terme du GS pour calculer les similarités
    for term, term_embedding in gs_embeddings.items():
        best_similarity = -1
        best_core_concept = 'o'  # Valeur par défaut si aucune similarité ne dépasse le seuil
        # Calculer la similarité cosinus avec chaque core concept
        for concept, concept_embedding in core_concepts_embeddings.items():
            similarity = cosine_similarity([term_embedding], [concept_embedding])[0, 0]

            # Vérifier si cette similarité est la plus élevée trouvée pour le terme
            if similarity > best_similarity:
                best_similarity = similarity
                best_core_concept = concept
        similarity.append(best_similarity)
        # Assigner la classe en fonction du seuil
        classifications[term] = {
            'core_concept': best_core_concept if best_similarity >= threshold else 'o',
            'class': best_core_concept if best_similarity >= threshold else 'o'
        }

    median = np.median(similarity)
    print(median)
    return classifications



def load_gold_standard_labels(gs_path):
    """
    Charge les labels du gold standard à partir d'un fichier Excel.

    :param gs_path: Chemin vers le fichier Excel contenant le GS
    :return: DataFrame avec les colonnes 'term' et 'label'
    """
    # Charger la feuille contenant les termes et labels
    gold_standard_df = pd.read_excel(gs_path, sheet_name='Sheet1')

    # Vérifier que les colonnes nécessaires sont présentes
    if 'term' not in gold_standard_df.columns or 'label' not in gold_standard_df.columns:
        raise ValueError("Le fichier GS doit contenir les colonnes 'term' et 'label'.")

    return gold_standard_df[['term', 'label']]


def evaluate_classification(gs_path, classifications):
    """
    Évalue la classification en calculant les métriques de performance.

    :param gs_path: Chemin vers le fichier Excel contenant le GS avec les termes et leurs labels
    :param classifications: Dictionnaire des classifications pour chaque terme
    :return: DataFrame des métriques d'évaluation
    """
    # Charger le fichier GS et obtenir les labels
    gold_standard_labels = load_gold_standard_labels(gs_path)

    # Extraire les labels réels et les prédictions, en les convertissant en chaînes de caractères
    true_labels = gold_standard_labels['label'].astype(str).values
    predicted_labels = [str(classifications[term]['core_concept']) for term in gold_standard_labels['term']]
    print("True labels (sample):", true_labels[:10])
    print("Predicted labels (sample):", predicted_labels[:10])
    # Calcul des métriques
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)

    # Stocker les métriques dans un dictionnaire pour les afficher sous forme de DataFrame
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

    metrics_df = pd.DataFrame([metrics], index=['Metrics'])
    return metrics_df








KG_path = config["KG_path"]
Gs_path = config["Gs_path"]

embeddings_dict,cc_embd = generate_gs_embeddings(KG_path,"checkpoints/model_epoch_20.pth",Gs_path,config["core_concepts"],config,embedding_model= "Bert")
# Paramètres de seuil de classification

thresholds = [0.1,0.2,0.4,0.5,0.6,0.7,0.8]
for threshold in thresholds:
        print("\n************** threshold = ", threshold, "**********************\n")
        # Classification des termes en fonction de la similarité cosinus avec les core concepts
        classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, threshold=threshold)

        # Évaluation de la classification
        metrics_df = evaluate_classification(Gs_path, classifications)

        # Afficher les résultats
        print(metrics_df)
# print(embeddings_dict.keys(), "\n",cc_embd.keys())

