import os
import pickle
import re
import sys

from torch_geometric.nn import GAE
from tqdm import tqdm
from transformers import BeitModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from data.GraphDataLoader import GraphDataLoader
from src.bert_embedding.BertEmbedder import BertEmbedder
from src.layers.Dismult import DistMultDecoder
from src.layers.GATDecoder import GATDecoder
from src.layers.GATEncoder import GATEncoder
from src.layers.GCNDecoder import GCNDecoder
from src.layers.GCNEncoder import GCNEncoder
from src.layers.MLPDecoder import MLPDecoder
from src.layers.TransGCNDecoder import TransGCNDecoder
from src.model.clustering import kmeans_classify_with_centroid_flag, dbscan_classify_with_centroid_flag_cosine, \
    kmeans_with_fixed_centroids


from GraphDataPreparation import GraphDataPreparation
from src.layers.TransGCNEncoder import TransGCNEncoder
from src.model.gnn_classifier.classifier_utils import  instantiate_encoder, instantiate_decoder
from MRGAE import MRGAE
from torch_geometric.nn import GAE
from ConvE import ConvE
from RGCNDecoder import RGCNDecoder
from RGCNEncoder import RGCNEncoder
from MRGAE import MRGAE
from utils.utils import save_model, load_model_checkpoint, load_gold_standard_labels, \
    save_model_with_hyperparams, set_seed, load_model_from_checkpoint
from config import config
import torch
from torch import optim
from torch_geometric.data import Data
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict

sys.path.append("../../utilities")
import utilities as u

import torch
import random
import numpy as np

import torch
import torch.nn.functional as F

import os

# Set the environment variable programmatically
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Enable deterministic algorithms in PyTorch
import torch
torch.use_deterministic_algorithms(True)

torch.manual_seed(42)
import numpy as np
np.random.seed(42)
import random
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False

def extract_params(filename):
    # Utilise une expression régulière pour extraire les valeurs de bases et channels
    match = re.search(r"bases(\d+)_channels(\d+)-(\d+)", filename)
    if match:
        bases = int(match.group(1))
        channels = [int(match.group(2)), int(match.group(3))]
        return bases, channels
    else:
        raise ValueError("Le format du nom de fichier n'est pas valide")


#### OLD !!!!  -----> to be updated and improved
def generate_gs_embeddings_full_batch(graph_path, checkpoint_path, gs_path, core_concepts, config, embedding_model = "GNN", emb_file = None,bert_model = "bert-base-uncased"):

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
    if embedding_model == "Bert":#:not emb_file:
        print("Bert")
        return generate_Bert_Embeddings(config["Entities_path"], gs_path, config["core_concepts"], model_name =  bert_model) ## best: "pritamdeka/S-BioBert-snli-multinli-stsb"; pritamdeka/S-PubMedBert-MS-MARCO; "sentence-transformers/all-MiniLM-L6-v2"

    # Charger le graphe avec GraphDataPreparation
    gdp = GraphDataPreparation(config["Entities_path"], graph_path,
                               edges_embd_path=config["Edges_path"], is_directed=True)
    data = gdp.prepare_graph_with_type()
    data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(config["device"])
    print(data)

    if embedding_model == "GNN" and not emb_file:
    # Initialisation du modèle
        RGCN_encoder = RGCNEncoder(data, config["out_channels"], config["num_layers"], config["num_bases"]).to(
            config["device"])
        RGCN_decoder = RGCNDecoder(RGCN_encoder, data, config["num_bases"], config["alpha"]).to(config["device"])
        config["convE_config"]["embedding_dim"] = config["out_channels"][-1]
        config["convE_config"]["hidden_size"] = config["coresp_hidden_sizes"][config["out_channels"][-1]]
        # r_decoder = ConvE(config["convE_config"])

        # GCN_encoder = GCNEncoder(data, config["out_channels"], config["num_layers"]).to(config["device"])
        # GCN_decoder = GCNDecoder(GCN_encoder, data, config["alpha"]).to(config["device"])

        # autoencoder = MC2GEA(GCN_encoder, GCN_decoder, projections = [config["out_channels"][-1], config["out_channels"][-1]]).to(config["device"])

        autoencoder = MRGAE(RGCN_encoder, RGCN_decoder,projections=[256, 256]).to(config["device"])

    # RGCN_encoder = RGCNEncoder1(data, config["out_channels"], config["num_layers"], config["num_bases"]).to(
        #     config["device"])
        # # RGCN_decoder = RGCNDecoder(RGCN_encoder, data, num_bases, config["alpha"]).to(device)
        # # autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder).to(device)
        # autoencoder = DeepGraphInfomax(
        #     hidden_channels = config["out_channels"][1], encoder=RGCN_encoder,
        #     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        #     corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index)
        # ).to(config["device"])



        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])



        model, optimizer, start_epoch = load_model_checkpoint(autoencoder, optimizer, checkpoint_path)
        save_model_with_hyperparams(model.encoder, optim.Adam(model.encoder.parameters(), lr=config["learning_rate"]),
                                     75,10,[640,512], "Encoder")

        model.to(config["device"])
        model.eval()
        # Encoder le graphe
        with torch.no_grad():
             print("\n-----GNN---------\n")

             embeddings = model.encode(data)
             #embeddings = model.encoder(data.x, data.edge_index)

            # embeddings_decode = model.decode_x(data,embeddings)
             # #    # checkpoint_path
             # u.save_to_pickle(checkpoint_path+".pickle", embeddings)
             # u.save_to_pickle("Recons_X_MSE_sim.pickle", embeddings_decode)
             # u.save_to_pickle("X.pickle", data.x)

            # embeddings = model.encoder(data.x, data.edge_index, data.edge_type)
            # print(embeddings[0])
    elif embedding_model == "Bert":#:not emb_file:
        print("-------------Bert--------------\n")
        embeddings = data.x
        # print(embeddings, print(embeddings.shape))
        # print(embeddings[0])
    elif emb_file:
        #embeddings = data.x
        #emb_file
        print("load_emb_from_file")
        embeddings = u.read_pickle_file( "results/embeddings_DGI.pkl")
        #print(embeddings, print(embeddings.shape))
        print(embeddings.shape)
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
            gs_embeddings[term] = embeddings[node_id].detach().cpu().numpy()  # Extraire l'embedding et le stocker

        # Vérifier si le terme est dans les core concepts
        if term_lower in [concept.lower() for concept in core_concepts]:
            core_concepts_embeddings[term] = embeddings[node_id].detach().cpu().numpy()
    #
    # print(f"Nombre de termes du GS avec embeddings : {len(gs_embeddings)}")
    # print(f"Nombre de core concepts avec embeddings : {len(core_concepts_embeddings)}")
    return gs_embeddings, core_concepts_embeddings

### New version
def generate_Bert_Embeddings(Entities_embd_path, gs_path, core_concepts, model_name="allenai/scibert"):
    from tqdm import tqdm
    import pandas as pd

    gs_terms = pd.read_excel(gs_path)['term'].tolist()
    gs_embeddings = {}
    core_concepts_embeddings = {}

    # Charger les embeddings déjà existants
    try:
        entities_embd = u.read_pickle_file(Entities_embd_path)
        if entities_embd is None:
            entities_embd = dict()
    except:
        entities_embd = dict()

    # Initialiser une seule instance de BertEmbedder
    be = BertEmbedder(model_name)

    print("GS Embedding ...\n")
    for term in tqdm(gs_terms):
        if term in entities_embd:
            gs_embeddings[term] = entities_embd[term].squeeze().detach().cpu().numpy()
        else:

            gs_embeddings[term] = be.embed_entity(term).squeeze().detach().cpu().numpy()

    print("CC Embedding ...\n")
    for cc in tqdm(core_concepts):
        if cc in entities_embd:
            core_concepts_embeddings[cc] = entities_embd[cc].squeeze().detach().cpu().numpy()
        else:

            core_concepts_embeddings[cc] = be.embed_entity(cc).squeeze().detach().cpu().numpy()

    return gs_embeddings, core_concepts_embeddings



def generate_gs_embeddgs_from_model_mini_batch(model, data,gs_path, core_concepts, gdp, config, is_encoder = False):
    model.eval()
    set_seed(42)
    data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()

    if is_encoder:
        with torch.no_grad():
            embeddings = torch.tensor([], dtype=torch.float32).to(config["device"])  # Initialize empty tensor
            for batch in data_loader:
                mask = torch.isin(batch.n_id, batch.input_id)
                batch_embeddings = model(data)
                masked_embd = batch_embeddings[mask]

                # Concatenate results instead of appending to a list
                embeddings = torch.cat((embeddings, masked_embd), dim=0)

    else:
        with torch.no_grad():
             # embeddings = model.encode(data)
            embeddings = torch.tensor([], dtype=torch.float32).to(config["device"])  # Initialize empty tensor
            for batch in data_loader:
                mask = torch.isin(batch.n_id, batch.input_id)
                batch_embeddings = model.encode(batch)
                masked_embd = batch_embeddings[mask]
                # Concatenate results instead of appending to a list
                embeddings = torch.cat((embeddings, masked_embd), dim=0)

    gs_df = pd.read_excel(gs_path, sheet_name='Sheet1')
    gs_terms = set(gs_df['term'].str.lower().unique())  # Termes du GS en minuscules pour uniformiser
    node_index_to_text = gdp.decode_indexes()
    gs_embeddings = {}
    core_concepts_embeddings = {}
    for node_id, term in node_index_to_text.items():
        term_lower = term.lower()
        if term_lower in gs_terms:
            gs_embeddings[term] = embeddings[node_id].detach().cpu().numpy()
        if term_lower in [concept.lower() for concept in core_concepts]:
            core_concepts_embeddings[term] = embeddings[node_id].detach().cpu().numpy()
    return gs_embeddings, core_concepts_embeddings


### old version !!!!!!!!
def generate_gs_embeddgs_from_model(model, data,gs_path, core_concepts, gdp,is_encoder = False):
    model.eval()
    if is_encoder:
        with torch.no_grad():
            embeddings = model(data)
    else:
        with torch.no_grad():
             embeddings = model.encode(data)
           # embeddings = model.encoder(data.x, data.edge_index, data.edge_type)
    gs_df = pd.read_excel(gs_path, sheet_name='Sheet1')
    gs_terms = set(gs_df['term'].str.lower().unique())  # Termes du GS en minuscules pour uniformiser
    node_index_to_text = gdp.decode_indexes()
    gs_embeddings = {}
    core_concepts_embeddings = {}
    for node_id, term in node_index_to_text.items():
        term_lower = term.lower()
        if term_lower in gs_terms:
            gs_embeddings[term] = embeddings[node_id].detach().cpu().numpy()
        if term_lower in [concept.lower() for concept in core_concepts]:
            core_concepts_embeddings[term] = embeddings[node_id].detach().cpu().numpy()
    return gs_embeddings, core_concepts_embeddings

##################### NEW Version of Embeddings generation from GNN
### New Version ---> Embeds only a given term
def generate_one_node_term_embedding(model, graph, gdp, term, config):
    model.eval()
    with torch.no_grad():
        term_index = gdp.nodes_index[term]

        # Crée un DataLoader avec un seul nœud en entrée
        data_loader = GraphDataLoader(
            graph,
            input_nodes=torch.tensor([term_index]),
            num_neighbors=config["num_neighbors"],
            batch_size=1,
            shuffle=False
        ).get_loader()
        batch = next(iter(data_loader))
        embeddings = model(batch)  # output shape: [num_nodes_in_batch, hidden_dim]
        mask = torch.isin(batch.n_id, torch.tensor([term_index], device=batch.n_id.device))
        node_embedding = embeddings[mask][0]  # Il ne devrait y avoir qu’un seul nœud
        return node_embedding


### New Version ---> Embeds a list of terms by batching

def generate_batch_term_embeddings(model, graph, gdp, terms, batch_size, num_neighbors, seed=42):
    set_seed(seed)
    model.eval()
    with torch.no_grad():
        # Convertit les termes en indices
        term_indices = []
        term_to_index = {}
        for term in terms:
            idx = gdp.nodes_index.get(term)
            if idx is not None:
                term_indices.append(idx)
                term_to_index[idx] = term
            else:
                print(f"Warning: term '{term}' not found in gdp index.")

        if not term_indices:
            return {}

        input_tensor = torch.tensor(term_indices, dtype=torch.long)
        # DataLoader avec tous les nœuds ciblés
        data_loader = GraphDataLoader(
            graph,
            input_nodes=term_indices,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False
        ).get_loader()
        embeddings_dict = {}
        for batch in tqdm(data_loader):
            if isinstance(model.encoder, TransGCNEncoder):
                batch_embeddings, _ = model.encode(batch)
            else:
                batch_embeddings = model.encode(batch)
            # batch_embeddings = model(batch)  # shape: [num_nodes_in_batch, dim]
            mask = torch.isin(batch.n_id, input_tensor[batch.input_id])

            input_embeddings = batch_embeddings[mask]
            embedding_indexes = batch.n_id[mask]
            for idx, emb in zip(embedding_indexes, input_embeddings):
                embeddings_dict[gdp.decode_indexes()[int(idx)]] = emb.detach().cpu().numpy()
        return embeddings_dict


### New Version ---> Embeds a list of terms from a given GS and CC
def generate_batch_GS_term_embeddings(model, graph, gdp, gs_path, core_concepts, batch_size = config['test_batch_size'], num_neighbors = config['num_neighbors'], seed=42):
    set_seed(seed)


    if gs_path == "whole_graph":
        GS_terms =  list(gdp.nodes_index.keys())
    else:
        GS_cs_pd = pd.read_excel(gs_path)

        GS_terms = GS_cs_pd.term.tolist()
    dict_terms_embeddings = generate_batch_term_embeddings(model, graph, gdp, GS_terms, batch_size, num_neighbors, seed)
    dict_cc_embeddings = generate_batch_term_embeddings(model, graph, gdp, core_concepts, batch_size, num_neighbors, seed)
    return dict_terms_embeddings, dict_cc_embeddings


def classify_terms_by_cosine_similarity(gs_embeddings, core_concepts_embeddings, with_other = False, threshold = 0.5, with_similarity = False):
    """
    Classe les termes du GS en fonction de leur similarité cosinus avec les core concepts.
    :param with_other: si on va considerer la classe 'other'
    :param gs_embeddings: Dictionnaire avec les termes du GS comme clés et leurs embeddings comme valeurs
    :param core_concepts_embeddings: Dictionnaire avec les core concepts comme clés et leurs embeddings comme valeurs
    :param threshold: Seuil de similarité cosinus pour la classification
    :return: Dictionnaire avec chaque terme du GS, son core concept le plus proche, et la classe prédite
    """

    similarities = []
    classifications = {}
    for term, term_embedding in gs_embeddings.items():
        best_similarity = -1
        best_core_concept = 'o'  # Valeur par défaut si aucune similarité ne dépasse le seuil
        for concept, concept_embedding in core_concepts_embeddings.items():
            similarity = cosine_similarity([term_embedding], [concept_embedding])[0, 0]

            # Vérifier si cette similarité est la plus élevée trouvée pour le terme
            if similarity > best_similarity:
                best_similarity = similarity
                best_core_concept = concept
        similarities.append(best_similarity)
        # Assigner la classe: 2 choix en utilisant la classe "other" pour ce qui sont loin (thrshold)
        if not with_other:
            class_ =  best_core_concept
        else:
            class_ = best_core_concept if best_similarity >= threshold else 'o'
        classifications[term] = {
            'class': class_
        }
        if with_similarity:
            classifications[term]['similarity'] = best_similarity
    # print(f'*****Median: {np.median(similarities)} ********** Mean: {np.mean(similarities)} ***********')
    return classifications


def evaluate_classification(gs_path, classifications, export_preds_path = None):
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
    terms = gold_standard_labels['term'].astype(str).values

    predicted_labels = [str(classifications[term]['class']) for term in gold_standard_labels['term']]
    # Calcul des métriques
    #
    # from sklearn.metrics import classification_report
    #
    # # Afficher un rapport de classification par classe
    # print(classification_report(true_labels, predicted_labels, zero_division=0))
    # exit(555)
    # count_p = Counter(predicted_labels)
    # count_b = Counter(true_labels)
    # print("\n+++++++++++ predicted +++++++++++\n",count_p ,"\n++++++++++++++++\n")
    # print("\n+++++++++++ Benchmark +++++++++++\n",count_b ,"\n++++++++++++++++\n")
    # print(type(predicted_labels), "\n", type(true_labels),"\n")
    #
    if export_preds_path:
        data__ = pd.DataFrame({
             "term": terms,
             "Predictions": predicted_labels,
             "Labels": true_labels
        })
        #
        # # Saving to an Excel file
        #
        file_path = export_preds_path
        data__.to_excel(file_path, index=False)

    print("\nRapport de Classification:\n")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }
    metrics_df = pd.DataFrame([metrics], index=['Metrics'])
    return metrics_df
## generate_batch_GS_term_embeddings(model, graph, gdp, gs_path, core_concepts, batch_size = config['test_batch_size'], num_neighbors = config['num_neighbors'], seed=42)


def evaluate(model, data,gs_path, core_concepts, gdp, config,is_encoder = False, export_preds_path = None):
    gs_embeddings, core_concepts_embeddings = generate_batch_GS_term_embeddings(model, data, gdp, gs_path, core_concepts,batch_size = config['test_batch_size'], num_neighbors = config['num_neighbors'], seed=42)
    # print(core_concepts_embeddings["operating system"])
    classifications = classify_terms_by_cosine_similarity(gs_embeddings, core_concepts_embeddings, with_other = False)
    metrics_df = evaluate_classification(gs_path, classifications, export_preds_path = export_preds_path)
    metrics = {
        'accuracy': metrics_df.loc['Metrics', 'accuracy'],
        'f1_score': metrics_df.loc['Metrics', 'f1_score'],
        'precision': metrics_df.loc['Metrics', 'precision'],
        'recall': metrics_df.loc['Metrics', 'recall']
    }

    return metrics




def assign_top_k_pseudo_labels_batched(model, data, core_concepts, gdp, config, top_k = 10, output_path = None):
    Node_embeddings, core_concepts_embeddings = generate_batch_GS_term_embeddings(model, data, gdp, "whole_graph", core_concepts,batch_size = config['test_batch_size'], num_neighbors = config['num_neighbors'], seed=42)
    # print(core_concepts_embeddings["operating system"])
    classifications = classify_terms_by_cosine_similarity(Node_embeddings, core_concepts_embeddings, with_other = False, with_similarity = True)
    top_k_by_class = defaultdict(list)
    for term, info in classifications.items():
        class_ = info['class']
        similarity = info['similarity']
        top_k_by_class[class_].append((term, similarity))

    # Construction du DataFrame final
    rows = []
    for label, terms in top_k_by_class.items():
        top_terms = sorted(terms, key=lambda x: x[1], reverse=True)[:top_k]
        for term, sim in top_terms:
            rows.append({'term': term, 'label': label, 'similarity': round(sim, 6)})

    df = pd.DataFrame(rows)

    # Sauvegarde Excel
    if output_path:
        df.to_excel(output_path, index=False)

    return df




## NEW
def evaluate_trained_GNN(model_path, config, export_preds_path = None):
    print(config["Entities_path"],'\n',config["Edges_path"],'\n',config['KG_path'])
    gdp = GraphDataPreparation(config["Entities_path"], config['KG_path'],
                               edges_embd_path=config["Edges_path"], is_directed=True)

    data = gdp.prepare_graph_with_type()
    data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(config["device"])
    print(data)
    model = load_model_from_checkpoint_(
        model_path, data, config, gdp)


    metrics = evaluate(model, data,config["Gs_path_no_other"], config["core_concepts"], gdp, config, export_preds_path = export_preds_path)

    return metrics




def evaluate_all(KG_path, GS_path, ckpt_dir, config, embedding_model = "GNN", with_other = True, thresholds_list = [0.5], emb_file = None, bert_model="bert-base-uncased", export_preds_path = None):

    if embedding_model ==  "Bert":



        embeddings_dict, cc_embd = generate_gs_embeddings_full_batch(KG_path,
                                                          "",
                                                          GS_path, config["core_concepts"], config,
                                                          embedding_model="Bert", emb_file=emb_file, bert_model= bert_model)

        if with_other:
            for threshold in thresholds_list:
                print(f'\n************* {threshold} *****************\n')
                classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, threshold=threshold,
                                                                      with_other=with_other)
                metrics_df = evaluate_classification(GS_path, classifications)
                print(metrics_df)
            return

        else:

            classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, with_other=with_other)
            metrics_df = evaluate_classification(GS_path, classifications, export_preds_path = export_preds_path)
            print(metrics_df)
            return metrics_df









    for filename in os.listdir(ckpt_dir):
            if filename.endswith(".pth"):
                if embedding_model != "GNN":
                    embeddings_dict, cc_embd = generate_gs_embeddings(KG_path,
                                                                      str(filename),
                                                                      GS_path, config["core_concepts"], config,
                                                                      embedding_model="Bert", emb_file = emb_file)

                    if with_other:
                        for threshold in thresholds_list:
                            print(f'\n************* {threshold} *****************\n')
                            classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, threshold=threshold,
                                                                                  with_other=with_other)
                            metrics_df = evaluate_classification(Gs_path, classifications)
                            print(metrics_df)
                        return
                    else:
                        classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, with_other=with_other)
                        metrics_df = evaluate_classification(Gs_path, classifications)
                        print(metrics_df)
                        return metrics_df

                else:

                        print("\n --------------------"+ filename + "-------------------------- \n")
                        bases, channels = extract_params(filename)
                        if bases is not None and channels is not None:
                            config['num_bases'] = bases
                            config["out_channels"] = channels
                            embeddings_dict, cc_embd = generate_gs_embeddings(KG_path,
                                                                              ckpt_dir+'/'+str(filename),
                                                                              GS_path, config["core_concepts"], config,
                                                                              embedding_model = embedding_model)
                            if with_other:
                                for threshold in thresholds_list:
                                    print(f'\n************* {threshold} *****************\n')
                                    classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, threshold=threshold, with_other=with_other)
                                    metrics_df = evaluate_classification(Gs_path, classifications)
                                    print(metrics_df)
                            else:
                                classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, with_other=with_other)
                                print(classifications)
                                metrics_df = evaluate_classification(Gs_path, classifications)
                                print(metrics_df)




def evaluate_all_save_best(KG_path, GS_path, ckpt_dir, config, embedding_model = "GNN", with_other = True, thresholds_list = [0.5]):
    best_results = {}
    for filename in os.listdir(ckpt_dir):
        if filename.endswith(".pth"):
            if embedding_model != "GNN":
                embeddings_dict, cc_embd = generate_gs_embeddings(KG_path,
                                                                  str(filename),
                                                                  GS_path, config["core_concepts"], config,
                                                                  embedding_model="bert")

                if with_other:
                    model_metrics = []
                    for threshold in thresholds_list:

                        classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, threshold=threshold,
                                                                              with_other=with_other)
                        metrics_df = evaluate_classification(Gs_path, classifications)
                        metrics = {
                            'threshold': threshold,
                            'accuracy': metrics_df.loc['Metrics', 'accuracy'],
                            'f1_score': metrics_df.loc['Metrics', 'f1_score'],
                            'precision': metrics_df.loc['Metrics', 'precision'],
                            'recall': metrics_df.loc['Metrics', 'recall']
                        }
                        model_metrics.append(metrics)
                    best_accuracy = max(model_metrics, key=lambda x: x['accuracy'])
                    best_f1 = max(model_metrics, key=lambda x: x['f1_score'])
                    if best_accuracy == best_f1:
                        best_results["BERT"] = {'best_result': best_accuracy}
                    else:
                        best_results["BERT"] = {
                            'best_accuracy': best_accuracy,
                            'best_f1': best_f1
                        }

                    return best_results
                else:
                    classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, with_other=with_other)
                    metrics_df = evaluate_classification(Gs_path, classifications)
                    metrics = {
                        'threshold': "-",
                        'accuracy': metrics_df.loc['Metrics', 'accuracy'],
                        'f1_score': metrics_df.loc['Metrics', 'f1_score'],
                        'precision': metrics_df.loc['Metrics', 'precision'],
                        'recall': metrics_df.loc['Metrics', 'recall']
                    }
                    best_results["BERT"] = {'best_result': metrics}
                    return best_results

            else:

                print("\n --------------------"+ filename + "-------------------------- \n")
                bases, channels = extract_params(filename)
                if bases is not None and channels is not None:
                    config['num_bases'] = bases
                    config["out_channels"] = channels
                    embeddings_dict, cc_embd = generate_gs_embeddings(KG_path,
                                                                      ckpt_dir+'/'+str(filename),
                                                                      GS_path, config["core_concepts"], config,
                                                                      embedding_model = embedding_model)
                    if with_other:
                        model_metrics = []
                        for threshold in thresholds_list:
                            print(f'\n************* {threshold} *****************\n')
                            classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, threshold=threshold, with_other=with_other)
                            metrics_df = evaluate_classification(Gs_path, classifications)
                            metrics = {
                                'threshold': threshold,
                                'accuracy': metrics_df.loc['Metrics', 'accuracy'],
                                'f1_score': metrics_df.loc['Metrics', 'f1_score'],
                                'precision': metrics_df.loc['Metrics', 'precision'],
                                'recall': metrics_df.loc['Metrics', 'recall']
                            }
                            model_metrics.append(metrics)
                        best_accuracy = max(model_metrics, key=lambda x: x['accuracy'])
                        best_f1 = max(model_metrics, key=lambda x: x['f1_score'])
                        if best_accuracy == best_f1:
                            best_results[filename] = {'best_result': best_accuracy}
                        else:
                            best_results[filename] = {
                                'best_accuracy': best_accuracy,
                                'best_f1': best_f1
                            }

                    else:
                        classifications = classify_terms_by_cosine_similarity(embeddings_dict, cc_embd, with_other=with_other)
                        metrics_df = evaluate_classification(Gs_path, classifications)
                        # print(metrics_df)
                        metrics = {
                            'threshold': "-",
                            'accuracy': metrics_df.loc['Metrics', 'accuracy'],
                            'f1_score': metrics_df.loc['Metrics', 'f1_score'],
                            'precision': metrics_df.loc['Metrics', 'precision'],
                            'recall': metrics_df.loc['Metrics', 'recall']
                        }
                        best_results[filename] = {'best_result': metrics}
    # # #

    return best_results

def parse_model_filename(filename):
    """
    Extrait les infos du nom de fichier.
    Ex: 'Recons_X_bases-10_channels_512-256_enc-RGCN_dec-MLP_best_acc.pth'
    """
    filename = filename.replace("_best_acc.pth", "").replace(".pth", "")
    pattern = r"(?P<task>.*?)_bases-(?P<bases>\d+)?_?channels_(?P<channels>[0-9\-]+)_enc-(?P<enc>\w+)_dec-(?P<dec>\w+)"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    task = match.group("task")
    bases = int(match.group("bases")) if match.group("bases") else None
    channels = list(map(int, match.group("channels").split('-')))
    encoder = match.group("enc")
    decoder = match.group("dec")

    return task, bases, channels, encoder, decoder

#
# def load_model_from_checkpoint_(checkpoint_path, data, config, gdp):
#     """
#     Charge un modèle depuis un checkpoint à partir du nom de fichier.
#     """
#     device = config["device"]
#     filename = checkpoint_path.split('/')[-1].replace("_best_acc.pth", "")
#     task, bases, out_channels, encoder_name, decoder_name = parse_model_filename(filename)
#     print(task, bases, out_channels, encoder_name, decoder_name)
#     msg_sens = config["message_sens"][0]
#
#     # --- Encoder ---
#     if encoder_name == "GCN":
#         encoder = GCNEncoder(data, out_channels, config["num_layers"], message_sens=msg_sens).to(device)
#     elif encoder_name == "RGCN":
#         encoder = RGCNEncoder(data, out_channels, config["num_layers"], bases, message_sens=msg_sens).to(device)
#     else:
#         raise ValueError(f"Unknown encoder: {encoder_name}")
#
#     # --- Decoder ---
#     if task == "Recons_X":
#         if decoder_name == "GCN":
#             decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
#         elif decoder_name == "RGCN":
#             decoder = RGCNDecoder(encoder, data, bases, config["alpha"], message_sens=msg_sens).to(device)
#         elif decoder_name == "MLP":
#             decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)
#         else:
#             raise ValueError(f"Unknown decoder: {decoder_name}")
#
#         model = MRGAE(encoder, decoder).to(device)
#
#     elif task == "Recons_A":
#         model = GAE(encoder).to(device)
#
#     elif task == "Recons_R":
#         r_decoder = DistMultDecoder(data.num_edge_types, out_channels[-1])
#         model = MRGAE(encoder, x_decoder=None, r_decoder=r_decoder).to(device)
#
#     elif task == "Double_reconstruction":
#         r_decoder = DistMultDecoder(data.num_edge_types, out_channels[-1])
#         if decoder_name == "GCN":
#             decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
#         elif decoder_name == "RGCN":
#             decoder = RGCNDecoder(encoder, data, bases, config["alpha"], message_sens=msg_sens).to(device)
#         elif decoder_name == "MLP":
#             decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)
#         else:
#             raise ValueError(f"Unknown decoder: {decoder_name}")
#         model = MRGAE(encoder, x_decoder=decoder, r_decoder=r_decoder).to(device)
#
#     else:
#         raise ValueError(f"Unknown task: {task}")
#
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     return model
#


def load_model_from_checkpoint_(checkpoint_path, data, config, gdp):
    """
    Charge un modèle depuis un checkpoint à partir du nom de fichier.
    """
    import torch
    import copy

    device = config["device"]
    filename = checkpoint_path.split('/')[-1].replace("_best_acc.pth", "")
    task, bases, out_channels, encoder_name, decoder_name = parse_model_filename(filename)
    print(task, bases, out_channels, encoder_name, decoder_name)
    msg_sens = config["message_sens"][0]

    # --- Encoder ---
    if encoder_name == "GCN":
        encoder = GCNEncoder(data, out_channels, config["num_layers"], message_sens=msg_sens).to(device)
    elif encoder_name == "RGCN":
        encoder = RGCNEncoder(data, out_channels, config["num_layers"], bases, message_sens=msg_sens).to(device)
    elif encoder_name in ["TransGCN_conv", "TransGCN_attn"]:
        encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                  kg_score_fn='TransE',
                                  variant='conv' if encoder_name.endswith("conv") else "attn",
                                  use_edges_info=config["use_edges_info"],
                                  activation='relu',
                                  bias=False).to(device)
    elif encoder_name in ["RotatEGCN_conv", "RotatEGCN_attn"]:
        encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                  kg_score_fn='RotatE',
                                  variant='conv' if encoder_name.endswith("conv") else "attn",
                                  use_edges_info=config["use_edges_info"],
                                  activation='relu',
                                  bias=False).to(device)
    elif encoder_name == "GAT":
        encoder = GATEncoder(data, out_channels, config["num_layers"]).to(device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    # --- Decoder ---
    decoder = None
    if task in ["Recons_X", "Double_reconstruction"]:
        if decoder_name == "GCN":
            decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
        elif decoder_name == "RGCN":
            decoder = RGCNDecoder(encoder, data, bases, config["alpha"], message_sens=msg_sens).to(device)
        elif decoder_name == "MLP":
            decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)
        elif decoder_name in ["TransGCN_conv", "TransGCN_attn"]:
            decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                      kg_score_fn='TransE',
                                      variant='conv' if decoder_name.endswith("conv") else "attn",
                                      use_edges_info=config["use_edges_info"]).to(device)
        elif decoder_name in ["RotatEGCN_conv", "RotatEGCN_attn"]:
            decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                      kg_score_fn='RotatE',
                                      variant='conv' if decoder_name.endswith("conv") else "attn",
                                      use_edges_info=config["use_edges_info"]).to(device)
        elif decoder_name == "GAT":
            decoder = GATDecoder(encoder, data, heads=4, alpha=0.01, dropout=0.3).to(device)
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")

    # --- Assemble Model ---
    if task == "Recons_X":
        model = MRGAE(encoder, decoder).to(device)
    elif task == "Recons_A":
        model = GAE(encoder).to(device)
    elif task == "Recons_R":
        r_decoder = DistMultDecoder(data.num_edge_types, out_channels[-1])
        model = MRGAE(encoder, x_decoder=None, r_decoder=r_decoder).to(device)
    elif task == "Double_reconstruction":
        r_decoder = DistMultDecoder(data.num_edge_types, out_channels[-1])
        model = MRGAE(encoder, x_decoder=decoder, r_decoder=r_decoder).to(device)
    else:
        raise ValueError(f"Unknown task: {task}")

    # --- Load Weights ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model












#
# KG_path = config["KG_path"]
# Gs_path = config["Gs_path_no_other"]
# thresholds_list = [0.6,0.7,0.8]

################################## Evaluate Bert/ sentence bert Medels ########################
# models_cs = [
#     # Sentence-BERT
#     "sentence-transformers/all-MiniLM-L6-v2",
#     "sentence-transformers/paraphrase-MiniLM-L6-v2",
#     # "sentence-transformers/paraphrase-mpnet-base-v2",
#     # "sentence-transformers/all-distilroberta-v1",
#     "sentence-transformers/nli-bert-base",
#     "sentence-transformers/nli-roberta-base-v2",
#     "sentence-transformers/msmarco-distilbert-base-v3",
#     "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
#     "sentence-transformers/stsb-roberta-base-v2",
#     "sentence-transformers/stsb-bert-base",
#
#     # Modèles scientifiques / biomédicaux
#     "allenai/scibert_scivocab_uncased",
#     "allenai/specter",
#     "pritamdeka/S-BioBert-snli-multinli-stsb",
#     "pritamdeka/S-PubMedBert-MS-MARCO",
#     "michiyasunaga/BioLinkBERT-base",
#     "dmis-lab/biobert-base-cased-v1.1",
#     "emilyalsentzer/Bio_ClinicalBERT",
#
#     # BERT classiques
#     "bert-base-uncased",
#     "bert-base-cased",
#     "roberta-base",
#     "distilbert-base-uncased",
#     "albert-base-v2",
#     "xlm-roberta-base",
#
#     # Modèles orientés code
#     "microsoft/codebert-base",
#     "microsoft/graphcodebert-base",
#     "Salesforce/codet5-base",
#     "facebook/bart-large"
# ]
#
# best_model = None
# best_accuracy = 0.0
# results = {}
#
# for model_name in models_cs:
#     print(f"Testing model: {model_name}")
#     try:
#         metrics = evaluate_all(
#             KG_path,
#             Gs_path,
#             "checkpoints/checkpoints_Recons_X_vf_75",
#             config,
#             embedding_model="Bert",
#             with_other=False,
#             thresholds_list=thresholds_list,
#             bert_model=model_name
#         )
#
#         accuracy = float(metrics["accuracy"].values[0])
#         f1 = float(metrics["f1_score"].values[0])
#         precision = float(metrics["precision"].values[0])
#         recall = float(metrics["recall"].values[0])
#
#         results[model_name] = {
#             "accuracy": accuracy,
#             "f1_score": f1,
#             "precision": precision,
#             "recall": recall
#         }
#
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model_name
#
#         print(f"Accuracy: {accuracy}")
#
#     except Exception as e:
#         print(f"Error with model {model_name}: {e}")
#
# # Export to Excel
# df_results = pd.DataFrame.from_dict(results, orient='index')
# df_results.index.name = "Model"
# df_results.to_excel("evaluation_results.xlsx")
#
# print("\n--- Summary ---")
# print(df_results)
# print(f"\nBest model: {best_model} with accuracy: {best_accuracy}")

#
#
#
#
#
#
#############################################################################################







#
# #
# metrics = evaluate_all(config['KG_path'], config["Gs_path_no_other"],None, config, embedding_model="Bert", with_other=False, bert_model= "sentence-transformers/all-MiniLM-L6-v2", export_preds_path= "classification_Bert.xlsx")
# print('\n----------------Bert------------------\n', metrics)

#
# metrics = evaluate_trained_GNN("best_models/recons_x/noisy/Recons_X_bases-5_channels_364-256_enc-RGCN_dec-MLP_best_acc.pth", config,"classification_noisy.xlsx")
# metrics = evaluate_trained_GNN("best_models/recons_x/clean/Recons_X_bases-10_channels_256-128_enc-RGCN_dec-MLP_best_acc.pth", config,"classification_clean.xlsx")
# print('\n---------------RGCN-----------------\n',metrics)
#

def generate_all_top_k_dataframes(model, data, core_concepts, gdp, config, k_list, output_base_path):
    """
    Génère les DataFrames des top-k pseudo-labels pour chaque k de k_list,
    sans recalculer les similarités à chaque fois (optimisé).

    :param model: modèle GNN
    :param data: graphe PyG
    :param core_concepts: concepts de référence
    :param gdp: préparateur de graphe
    :param config: dict avec test_batch_size et num_neighbors
    :param k_list: liste des valeurs de k (ex. [5, 10, 15])
    :param output_base_path: chemin de base pour sauvegarde .xlsx
    :return: dict {k: DataFrame}
    """
    import os
    import pandas as pd
    from collections import defaultdict

    # 1. Générer embeddings une seule fois
    Node_embeddings, core_concepts_embeddings = generate_batch_GS_term_embeddings(
        model,
        data,
        gdp,
        "whole_graph",
        core_concepts,
        batch_size=config['test_batch_size'],
        num_neighbors=config['num_neighbors'],
        seed=42
    )

    # 2. Calculer les similarités une seule fois
    classifications = classify_terms_by_cosine_similarity(
        Node_embeddings,
        core_concepts_embeddings,
        with_other=False,
        with_similarity=True
    )

    # 3. Grouper tous les résultats par type
    top_k_by_class = defaultdict(list)
    for term, info in classifications.items():
        label = info['class']
        sim = info['similarity']
        top_k_by_class[label].append((term, sim))

    # 4. Générer tous les DataFrames à partir des top-k filtrés
    base, ext = os.path.splitext(output_base_path)
    results = {}

    for k in k_list:
        rows = []
        for label, term_list in top_k_by_class.items():
            top_terms = sorted(term_list, key=lambda x: x[1], reverse=True)[:k]
            for term, sim in top_terms:
                rows.append({'term': term, 'label': label, 'similarity': round(sim, 6)})

        df = pd.DataFrame(rows)
        results[k] = df

        # Sauvegarde
        output_path_k = f"{base}_k{k}{ext}"
        df.to_excel(output_path_k, index=False)

    return results


#
# model_1 = "best_models/recons_x/noisy/Recons_X_bases-5_channels_364-256_enc-RGCN_dec-MLP_best_acc.pth"
# model_2 = "best_models/recons_x/noisy/Recons_X_channels_256-128_enc-RotatEGCN_conv_dec-MLP_best_acc.pth"
# model_3 = "best_models/recons_x/noisy/Recons_X_channels_256-128_enc-TransGCN_conv_dec-RotatEGCN_attn_best_acc.pth"
#
# gdp = GraphDataPreparation(config["Entities_path"], KG_path,
#                                edges_embd_path=config["Edges_path"], is_directed=True)
#
# data = gdp.prepare_graph_with_type()
# data = data.to(config["device"])
# print(data)
# model, model_info, checkpoint = load_model_from_checkpoint(model_3, data)
#
# print(model)
#
# output_base_path = "Recons_X_channels_384-256_enc-TransGCN_attn_dec-RotatEGCN_attn_best_acc.xlsx"
# results = generate_all_top_k_dataframes(model, data, config["core_concepts"], gdp, config,
#                                         [25,50,100,200,500,1000,1200,1500],
#                                         output_base_path)


# metrics = assign_top_k_pseudo_labels_batched(model,
#                                              data,
#                                              config['core_concepts'],
#                                              gdp,
#                                              config,
#                                              top_k=1000,
#                                              output_path="PL1000_v2_Recons_X_channels_256-128_enc-TransGCN_conv_dec-RotatEGCN_attn_best_acc.xlsx")
# print(metrics)








# # evaluate_all(KG_path, Gs_path, "checkpoints/Recons_X", config, embedding_model = "GNN", with_other = False, thresholds_list = thresholds_list, emb_file = None)




#
# gs_embd, cc_embd = generate_Bert_Embeddings(config["Entities_path"], Gs_path, config["core_concepts"], model_name = "allenai/scibert_scivocab_uncased" )
# # res_kmeans  = kmeans_classify_with_centroid_flag(gs_embd, len(config["core_concepts"]))
# res_kmeans  = kmeans_with_fixed_centroids(gs_embd, cc_embd)

# res_kmeans  = dbscan_classify_with_centroid_flag_cosine(gs_embd)
















# gdp = GraphDataPreparation(config["Entities_path"], KG_path,
#                                edges_embd_path=config["Edges_path"], is_directed=True)
#
# data = gdp.prepare_graph_with_type()
# data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(config["device"])
# print(data)
#
#
# # Initialisation du modèle
# RGCN_encoder = RGCNEncoder(data, config["out_channels"], config["num_layers"], config["num_bases"]).to(
#     config["device"])
# RGCN_decoder = RGCNDecoder(RGCN_encoder, data, config["num_bases"], config["alpha"]).to(config["device"])
# # config["convE_config"]["embedding_dim"] = config["out_channels"][1]
# # config["convE_config"]["hidden_size"] = config["coresp_hidden_sizes"][config["out_channels"][1]]
# # r_decoder = ConvE(config["convE_config"])
# config['num_bases'] = 10
# config["out_channels"] = [640,512]
#
# config["convE_config"]["embedding_dim"] = config["out_channels"][1]
# config["convE_config"]["hidden_size"] = config["coresp_hidden_sizes"][config["out_channels"][1]]
# r_decoder = ConvE(config["convE_config"])
#
#
# autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder, r_decoder = r_decoder).to(config["device"])
#
#
# optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
#
#
# # Charger le modèle et l'optimiseur à partir du checkpoint
# model_, optimizer, start_epoch = load_model_checkpoint(autoencoder, optimizer, "checkpoints/ckpt_expriments_MSE/best_model_bases10_channels640-512.pth")
#

# model.eval()

#
# torch.manual_seed(42)
#
# gs_embeddings, core_concepts_embeddings =  generate_gs_embeddings(KG_path, "checkpoints/ckpt_expriments_MSE/best_model_bases10_channels640-512.pth", Gs_path, config["core_concepts"], config, embedding_model = "GNN", emb_file = None)
# gs_embeddings_m, core_concepts_embeddings_m =generate_gs_embeddgs_from_model(model, data,Gs_path, config["core_concepts"], gdp)
# clss = classify_terms_by_cosine_similarity(gs_embeddings, core_concepts_embeddings, with_other = False, threshold = 0.5)
# calss_m = classify_terms_by_cosine_similarity(gs_embeddings_m, core_concepts_embeddings_m, with_other = False, threshold = 0.5)
#
# print(evaluate_classification(Gs_path, calss_m))
#
# print(evaluate(model_, data,Gs_path, config["core_concepts"], gdp))
#
# metrics = evaluate(model_, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
# print(metrics)
# print(clss)
# print(calss_m)
# for k in clss.keys():
#     if clss[k]["class"] !=  calss_m[k]["class"]:
#         print("!!!!!!!!")

# for cc in config["core_concepts"]:
#     #print(torch.tensor(core_concepts_embeddings[cc]) == torch.tensor(core_concepts_embeddings_m[cc]))
#     print(torch.equal(torch.tensor(core_concepts_embeddings[cc]), torch.tensor(core_concepts_embeddings_m[cc])))
# # evaluate(model, data, config["Gs_path"], config["core_concepts"], gdp)

# for k in gs_embeddings.keys():
#     if not torch.equal(torch.tensor(gs_embeddings[k]), torch.tensor(gs_embeddings[k])):
#         print("!!!")

# rows = []
# for model_name, metrics in res.items():
#     for metric_type, values in metrics.items():
#         row = {
#             "Model Name": model_name,
#             "Threshold": values["threshold"],
#             "Accuracy": values["accuracy"],
#             "F1-score": values["f1_score"],
#             "Precision": values["precision"],
#             "Recall": values["recall"]
#         }
#         rows.append(row)
#
# # Creating the DataFrame
# df_ = pd.DataFrame(rows)
# file_path = 'Recons_x_encoder_results.xlsx'
# df_.to_excel(file_path, index=False)
# # print(df_)