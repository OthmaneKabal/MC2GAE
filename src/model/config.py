import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", #
    "seed":0,
    "num_layers": 2,
    "alpha": 0.01,
    "max_masking_percentage": 0.3,
    "total_drop_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 256,
    "test_batch_size":256,
    # "cosine_loss_weight": 0.5,
    "shuffle": False,
    "num_neighbors": [200,200],
    "num_epochs": 50,
    "kg_score_fn":'TransE',
    "variant":'conv',
    "use_edges_info":True,
    "Entities_path": "../outputs/../outputs/EntitiesBertEmbedding_noicy_nci.pickle", #EntitiesBertEmbedding_NCI.pickle",../outputs/EntitiesBertEmbedding_noicy_nci
    "Edges_path": "../outputs/PredicatesBertEmbedding_noicy_nci.pickle",#PredicatesBertEmbedding_NCI.pickle",PredicatesBertEmbedding_noicy_nci

    "Entities_path_cs": "../outputs/EntitiesBertEmbeddingAugmented_cs__.pickle",
    "Edges_path_cs": "../outputs/PredicatesBertEmbeddingAugmented_cs__.pickle",
    "KG_path": "../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",   #KG_NCI_vf.json",MM_mapped_nci_KG

    "Gs_path_no_other": "../../data/UMLS/MM_mapped_nci_GS.xlsx",#nci_mm_GS_vf
     "KG_path_cs": "../../data/augmented_graph/augmented_graph_is_rules.json",
    "dataset": "umls_noisy_mapped",

    ####
    "Gs_path_no_other_cs": "../../data/GS_vf.xlsx",
    # "Gs_path_no_other": "../../data/GS_communs_terms_nci_mm_test.xlsx",
    ####
    "core_concepts_" : ['data structure',
                         'cryptography',
                         'software engineering',
                         'computer graphic',
                         'network security',
                         'computer programming',
                         'operating system',
                         'distributed computing',
                         'machine learning'
                        ],
    "core_concepts" : [
    "Body Part, Organ, or Organ Component",
    "Disease or Syndrome",
    "Finding",
    "Intellectual Product",
    "Laboratory Procedure",
    "Organic Chemical",
    "Pharmacologic Substance",
    "Therapeutic or Preventive Procedure"
],

    "training_task" : ["Recons_R"],
    "hyperparams_grid" : {"num_bases": [5,10], "out_channels": [[384,256],[512,364]]}, ## , [256,128], [128,64],[384,256], [64,32]
    "wandb_project_name": "Experiments_Recons_R_mapped_rel_noisy",
    "encoders": ["RGCN"],#"TransGCN_conv","TransGCN_attn","RotatEGCN_attn","RotatEGCN_conv", "GCN","GAT"],## "TransGCN_conv","TransGCN_attn","RotatEGCN_attn","RotatEGCN_conv", "GCN","RGCN","GAT"
    "decoders": ["MLP"],#,"TransGCN_attn","RotatEGCN_attn","RotatEGCN_conv", "GCN","RGCN","GAT","MLP"], ## "TransGCN_conv","TransGCN_attn","RotatEGCN_conv","RotatEGCN_attn"

    "message_sens": ["source_to_target"],
    "projections": None,
    "root_save_dir": "ckpt_Recons_R_mapped_noisy",
    "param_combinations": [{"encoder": "GCN","decoder":"GCN","out_channels":[640,512]},
                           {"encoder": "RGCN","decoder":"RGCN","out_channels":[640,512]},
                           {"encoder": "RGCN","decoder":"MLP","out_channels":[640,512]}
                           ],
    "convE_config": {
        'embedding_dim': 512,
        'input_drop': 0.2,
        'hidden_drop': 0.3,
        'feat_drop': 0.2,
        'embedding_shape1': 32,  # Nouvelle valeur
        'hidden_size': 27776,  # ## 2048 ----> 123008 ; 768 ----> 43648 ; 256 ----> 11904; 512 --> 27776 ; 128----> 3968
        'label_smoothing': 0.1,
        'use_bias': True,
    },
    "coresp_hidden_sizes": {768: 43648, 512: 27776, 256: 11904, 128: 3968, 64: 27776},
####### Mappers for Autopath #################
    "edges_path_map": {
        "umls_clean": "../outputs/umls_nci_bert_embeddings/PredicatesBioLinkBERT-base_UMLS_clean.pickle",
        "umls_noisy": "../outputs/umls_nci_bert_embeddings/PredicatesBioLinkBERT-base_UMLS_noisy.pickle", # PredicatesBertEmbedding_sc_L6-v2_noisy_nci.pickle,
        "umls_mitigated": "../outputs/PredicatesBertEmbedding_augmented_noicy_nci.pickle",
        "CS": "../outputs/umls_nci_bert_embeddings/Predicates_tintybert-cs.pickle",
        "CS_augmented":"../outputs/PredicatesBertEmbedding_sc_augmented.pickle",
        "umls_noisy_mapped":"../outputs/umls_nci_bert_embeddings/Predicates_mapped_rel_SM.pickle"
        },
    "entities_path_map": {
        "umls_clean": "../outputs/umls_nci_bert_embeddings/EntitiesBioLinkBERT-base_UMLS_clean.pickle",
        "umls_noisy": "../outputs/umls_nci_bert_embeddings/EntitiesBertEmbedding_sc_L6-v2_noisy_nci.pickle",
        "umls_mitigated": "../outputs/EntitiesBertEmbedding_augmented_noicy_nci.pickle",
        "CS":"../outputs/umls_nci_bert_embeddings/Entities_tintybert-cs.pickle", #"EntitiesBertEmbedding_sc.pickle",
        "CS_augmented":"../outputs/EntitiesBertEmbedding_sc_augmented.pickle",
        "umls_noisy_mapped": "../outputs/umls_nci_bert_embeddings/EntitiesBertEmbedding_sc_L6-v2_noisy_nci.pickle"
        },
    "kg_path_map" : {
    "umls_clean": "../../data/UMLS/clean/KG_NCI_vf.json",
    "umls_noisy": "../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",
    "umls_mitigated": "../../data/UMLS/noisy/augmented/is_a_augmented_MM_mapped_nci_All_R_KG.json",
    "CS": "../../data/Cs/org/original_graph_vf.json",
    "CS_augmented": "../../data/Cs/augmented/augmented_graph_is_rules.json",
    "umls_noisy_mapped": "../../data/UMLS/noisy/relations_mapped/kg_with_mapped_relations_cleaned.json",
        },
    "GS_path_map" : {
    "umls_clean": "../../data/UMLS/MM_mapped_nci_GS.xlsx",
    "umls_noisy": "../../data/UMLS/MM_mapped_nci_GS.xlsx",
    "umls_mitigated": "../../data/UMLS/MM_mapped_nci_GS.xlsx",
    "umls_noisy_mapped": "../../data/UMLS/MM_mapped_nci_GS.xlsx",
    "CS": "../../data/Cs/GS_vf.xlsx",
    "CS_augmented": "../../data/Cs/GS_vf.xlsx"
        },
    "train_set_path": {
        "umls_clean": "../../data/added_is_types/added_is_types_clean.xlsx",
        "umls_noisy": "../../data/added_is_types/added_is_types_noisy.xlsx"
    }
}

config["KG_path"] = config["kg_path_map"].get(
config["dataset"])
config["Gs_path_no_other"] = config["GS_path_map"].get(
config["dataset"])
config["Edges_path"] = config["edges_path_map"].get(config["dataset"])
config["Entities_path"] = config["entities_path_map"].get(config["dataset"])
config["train_set_path"] = config["train_set_path"].get(config["dataset"])





