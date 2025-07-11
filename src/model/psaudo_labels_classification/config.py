import torch
classifier_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", #
    "seed":42,
    "train_batch_size": 512,
    "test_batch_size": 512,
    "shuffle": True,
    "num_neighbors": [200, 200],
    "epochs":100,
    "learning_rate": 0.001,
    "MLP_layers": [128],
    "model_path_1":"../checkpoints/UMLS/noisy/org/Recons_X_channels_256-128_enc-TransGCN_conv_dec-RotatEGCN_attn_best_acc.pth",
    "model_path":"../checkpoints/UMLS/noisy/org/Recons_X_channels_384-256_enc-TransGCN_attn_dec-MLP_best_acc.pth",
    "model_path_3":"../checkpoints/UMLS/noisy/org/Recons_X_channels_384-256_enc-TransGCN_attn_dec-RotatEGCN_attn_best_acc.pth",
    "dataset": "umls_noisy",






######## Mappers for Autopath #################
    "edges_path_map": {
        "umls_clean": "../../outputs/PredicatesBertEmbedding_NCI.pickle",
        "umls_noisy": "../../outputs/PredicatesBertEmbedding_noicy_nci.pickle"
        },
    "entities_path_map": {
        "umls_clean": "../../outputs/EntitiesBertEmbedding_NCI.pickle",
        "umls_noisy": "../../outputs/EntitiesBertEmbedding_noicy_nci.pickle"
        },
    "kg_path_map" : {
    "umls_clean": "../../../data/UMLS/clean/KG_NCI_vf.json",
    "umls_noisy": "../../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",
    "CS": "CS/original_graph_vf.json"
        },
    "GS_path_map" : {
    "umls_clean": "../../../data/UMLS/MM_mapped_nci_GS.xlsx",
    "umls_noisy": "../../../data/UMLS/MM_mapped_nci_GS.xlsx",
    "CS": "CS/GS_vf.xlsx"
        },
    "train_set_path": {
        "umls_clean": "../../../data/added_is_types/added_is_types_clean.xlsx",
        "umls_noisy": "PL/Recons_X_channels_384-256_enc-TransGCN_attn_dec-MLP/1000.xlsx"
    }


}

classifier_config["kg_path"] = classifier_config["kg_path_map"].get(
    classifier_config["dataset"])
classifier_config["GS"] = classifier_config["GS_path_map"].get(
    classifier_config["dataset"])

classifier_config["edges_path"] = classifier_config["edges_path_map"].get(classifier_config["dataset"])
classifier_config["entities_path"] = classifier_config["entities_path_map"].get(classifier_config["dataset"])
classifier_config["train_set_path"] = classifier_config["train_set_path"].get(classifier_config["dataset"])
