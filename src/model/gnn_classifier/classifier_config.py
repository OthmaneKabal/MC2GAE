import torch
classifier_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", #
    "seed":42,
    "train_batch_size": 512,
    "test_batch_size": 512,
    "shuffle": True,
    "num_neighbors": [200, 200],
    "epochs":200,
    "learning_rate": 0.001,
    "classifier_encoder": "TransGCN_attn",
    "num_bases": 5,
    "num_layers": 2,
    "MLP_layers": [],
    "encoder_out_channels": [256,256],
    "message_sens": "source_to_target",
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
        "umls_noisy": "../../../data/added_is_types/added_is_types_noisy.xlsx"
    }


}

classifier_config["kg_path"] = classifier_config["kg_path_map"].get(
    classifier_config["dataset"])
classifier_config["GS"] = classifier_config["GS_path_map"].get(
    classifier_config["dataset"])

classifier_config["edges_path"] = classifier_config["edges_path_map"].get(classifier_config["dataset"])
classifier_config["entities_path"] = classifier_config["entities_path_map"].get(classifier_config["dataset"])
classifier_config["train_set_path"] = classifier_config["train_set_path"].get(classifier_config["dataset"])
