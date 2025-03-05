import os
import pickle
import re
import sys

from transformers import BeitModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from GraphDataPreparation import GraphDataPreparation
from ConvE import ConvE
from RGCNDecoder import RGCNDecoder
from RGCNEncoder import RGCNEncoder
from MRGAE import MRGAE
from utils.utils import save_model, load_model_checkpoint, load_gold_standard_labels, \
    save_model_with_hyperparams
from config import config
import torch
from torch import optim
from torch_geometric.data import Data
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


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




def generate_gs_embeddings(graph_path, checkpoint_path, gs_path, core_concepts, config, embedding_model = "GNN", emb_file = None):

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

        autoencoder = MRGEA(RGCN_encoder, RGCN_decoder,projections=[256, 256]).to(config["device"])

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

def evaluate(model, data,gs_path, core_concepts, gdp, is_encoder = False):
    gs_embeddings, core_concepts_embeddings = generate_gs_embeddgs_from_model(model, data,gs_path, core_concepts, gdp, is_encoder = is_encoder)
    # print(core_concepts_embeddings["operating system"])
    classifications = classify_terms_by_cosine_similarity(gs_embeddings, core_concepts_embeddings, with_other = False)
    metrics_df = evaluate_classification(gs_path, classifications)
    metrics = {
        'accuracy': metrics_df.loc['Metrics', 'accuracy'],
        'f1_score': metrics_df.loc['Metrics', 'f1_score'],
        'precision': metrics_df.loc['Metrics', 'precision'],
        'recall': metrics_df.loc['Metrics', 'recall']
    }

    return metrics

def classify_terms_by_cosine_similarity(gs_embeddings, core_concepts_embeddings, with_other = False, threshold = 0.5):
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
    # print(f'*****Median: {np.median(similarities)} ********** Mean: {np.mean(similarities)} ***********')
    return classifications





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
    # data__ = pd.DataFrame({
    #      "term": terms,
    #      "Predictions": predicted_labels,
    #      "Labels": true_labels
    # })
    # #
    # # # Saving to an Excel file
    # #
    # file_path = "classifications_Bert.xlsx"
    # data__.to_excel(file_path, index=False)
    #
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

def extract_params(filename):
    # Utilise une expression régulière pour extraire les valeurs de bases et channels
    match = re.search(r"bases(\d+)_channels(\d+)-(\d+)", filename)
    if match:
        bases = int(match.group(1))
        channels = [int(match.group(2)), int(match.group(3))]
        return bases, channels
    else:
        raise ValueError("Le format du nom de fichier n'est pas valide")



def evaluate_all(KG_path, GS_path, ckpt_dir, config, embedding_model = "GNN", with_other = True, thresholds_list = [0.5], emb_file = None):

    if embedding_model ==  "Bert":
        embeddings_dict, cc_embd = generate_gs_embeddings(KG_path,
                                                          "",
                                                          GS_path, config["core_concepts"], config,
                                                          embedding_model="Bert", emb_file=emb_file)

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
            return









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
                        return

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


KG_path = config["KG_path"]
Gs_path = config["Gs_path_no_other"]
thresholds_list = [0.6,0.7,0.8]



# evaluate_all(KG_path, Gs_path, "checkpoints/checkpoints_Recons_X_vf_75", config, embedding_model = "Bert", with_other = False, thresholds_list = thresholds_list)
# evaluate_all(KG_path, Gs_path, "new_data_RGCN", config, embedding_model = "GNN", with_other = False, thresholds_list = thresholds_list, emb_file = None)

















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