import os
import re
import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", #
    "seed": [0],
    "num_layers": 2,
    "alpha": 0.01,
    "max_masking_percentage": 0.3,
    "total_drop_rate": 0,
    "learning_rate": 0.001,
    "batch_size": 256,
    "test_batch_size":256,
    # "cosine_loss_weight": 0.5,
    "shuffle": False,
    "num_neighbors": [-1,-1],
    "num_epochs": 50,
    "num_steps": None,
    "kg_score_fn":'TransE',
    "variant":'conv',
    "use_edges_info":True,
    "plm_embedding_model": "sentence-transformers/all-MiniLM-L6-v2", ##"michiyasunaga/BioLinkBERT-base"
    "Entities_path": "../outputs/../outputs/EntitiesBertEmbedding_noicy_nci.pickle", #EntitiesBertEmbedding_NCI.pickle",../outputs/EntitiesBertEmbedding_noicy_nci
    "Edges_path": "../outputs/PredicatesBertEmbedding_noicy_nci.pickle",#PredicatesBertEmbedding_NCI.pickle",PredicatesBertEmbedding_noicy_nci

    "Entities_path_cs": "../outputs/EntitiesBertEmbeddingAugmented_cs__.pickle",
    "Edges_path_cs": "../outputs/PredicatesBertEmbeddingAugmented_cs__.pickle",
    "KG_path": "../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",   #KG_NCI_vf.json",MM_mapped_nci_KG

    "Gs_path_no_other": "../../data/UMLS/common_nodes.xlsx",#nci_mm_GS_vf
     "KG_path_cs": "../../data/augmented_graph/augmented_graph_is_rules.json",
    "dataset": "gt2kg_mapped_and_old_rel_norm",
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

    "training_task" : ["Recons_R_with_onto"],
    "graphmae_mask_rate": 0.3,
    "graphmae_replace_rate": 0.0,
    "graphmae_loss_fn": "SCE",
    "graphmae_sce_alpha": 3,
    "graphmae_decoder_remask": True,
    "graphmae_structure_masking": "random",  # "random", "pagerank", "degree", or "learnable"
    "graphmae_structure_alpha": 1.0,
    "graphmae_structure_schedule": "linear",  # "linear", "root", "geometric", or "constant"
    "graphmae_learnable_scorer_hidden": None,
    "recons_x_feature_masking": True,
    "run_linear_probe_on_best_loss": False,
    "linear_probe_gs_path": "../../data/UMLS/common_nodes.xlsx",
    "linear_probe_splits_dir": "../../data/UMLS/splits/umls_kg_splits",
    "linear_probe_split_seeds": [42, 123, 456, 789, 2024],
    "linear_probe_epochs": 300,
    "linear_probe_lr": 0.01,
    "linear_probe_weight_decay": 0.0,
    "linear_probe_patience": 50,
    "recons_r_training_mode": "all_batch_edges",  # "random_masked_only"/"removed_only" reconstruct only randomly masked edges; "mapped_only" hides mapped edges; "mapped_visible" keeps mapped edges visible; "all_batch_edges" is previous behavior
    "recons_r_target_relation_field": "predicate",  # "predicate" or "old_predicate" for mapping-guided controls
    "mapped_random_dynamic_mapped_fraction": 0.5,  # fraction of the masked edge budget sampled from is_mapped=True edges
    "mapped_only_dynamic_rate": 0.5,  # fraction of is_mapped=True edges masked each epoch for mapped_only_dynamic_* modes
    "mapped_mix_mapped_rate": 0.5,  # fraction of is_mapped=True edges masked each epoch for mapped_mix_dynamic_* modes
    "mapped_mix_non_mapped_rate": 0.5,  # fraction of is_mapped=False edges masked each epoch for mapped_mix_dynamic_* modes
    "all_mapped_plus_non_mapped_rate": 0.1,  # mask all mapped edges plus this fraction of non-mapped edges each epoch
    "edge_curriculum_split_ratio": 0.5,  # fraction of masked edges selected by model confidence; the rest is random
    "edge_curriculum_initial_rate": 0.05,
    "edge_curriculum_schedule": "linear",  # "linear", "root", "geometric", or "constant"
    "onto_KG_path": "../../data/UMLS/noisy/org/SM_network.json",
    "onto_entities_path": "outputs/SM_network/sentence-transformers_all-MiniLM-L6-v2_entities.pickle",
    "onto_edges_path": "outputs/SM_network/sentence-transformers_all-MiniLM-L6-v2_predicates.pickle",
    "lambda_onto": 0,
    "lambda_align": 0,
    "relation_alignment_loss": "mse",
    "lambda_core_contrastive": 0.0,
    "core_contrastive_temperature": 0,
    "lambda_core_align": 0,
    "core_alignment_loss": "mse",
    "domain_range_constraints_path": "../../data/UMLS/noisy/org/SM_domain_range_constraints.json",
    "lambda_domain_range": 0,
    "domain_range_temperature": 0,
    "lambda_onto_hierarchy": 0,
    "negative_sampling_mode": "uniform",  # "uniform" keeps the previous sampler
    "negative_corruption_mode": "entity_only",  # "mixed", "relation_only", or "entity_only" for KG negatives
    "negative_entity_sampling_scope": "batch",  # "batch" keeps old behavior; "global" samples corrupted entities from the full graph
    "kg_negative_sampling_seed": None,    # optional ablation: set to 0/1/42 while keeping "seed" fixed
    "track_kg_negative_sampling": False,  # set this to True to track the negative sampling distribution and save it to a file
    "kg_negative_tracking_dir": "analysis/negative_sampling",
    "kg_negative_tracking_max_examples": None,  # None saves all detailed positive/negative examples
    "track_onto_negative_sampling": False,  # set this to True to track ontology negative sampling per epoch
    "onto_negative_tracking_dir": "analysis/negative_sampling",
    "onto_negative_tracking_max_examples": None,
    "replay_onto_negative_sampling": False,  # set True to reuse saved ontology negatives from onto_negative_replay_path
    "onto_negative_replay_path": "analysis/negative_sampling/Recons_R_with_onto_channels_384-384_enc-RotatEGCN_attn_Dismult_seed0_onto_mixed.jsonl",
    "replay_kg_negative_sampling": False,  # set True to reuse saved KG negatives from kg_negative_replay_path
    "kg_negative_replay_path": "analysis/negative_sampling/Recons_R_with_onto_channels_384-384_enc-RotatEGCN_attn_Dismult_seed0_entity_only.jsonl",
    "soft_type_negative_ratio": 0.15,
    "soft_type_top_k": 5000,
    "soft_type_temperature": 0.7,
    "hyperparams_grid" : {"num_bases": [5,10], "out_channels": [[384,384]]}, ## , [256,128], [128,64],[384,256], [64,32]
    "wandb_project_name": None,
    "encoders": ["RotatEGCN_attn"],#"TransGCN_conv","TransGCN_attn","RotatEGCN_attn","RotatEGCN_conv", "GCN","GAT"],## "TransGCN_conv","TransGCN_attn","RotatEGCN_attn","RotatEGCN_conv", "GCN","RGCN","GAT"
    "decoders": ["MLP"],#,"TransGCN_attn","RotatEGCN_attn","RotatEGCN_conv", "GCN","RGCN","GAT","MLP"], ## "TransGCN_conv","TransGCN_attn","RotatEGCN_conv","RotatEGCN_attn"
    "message_sens": ["source_to_target"],
    "projections": None,
    "root_save_dir": None,
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
        'label_smoothing': 10,
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
        "umls_noisy_mapped":"../outputs/umls_nci_bert_embeddings/Predicates_mapped_rel_SM.pickle",
        "gt2kg_mapped_and_old_rel_norm": "outputs/GT2KG_mapped_and_old_rel_norm/sentence-transformers_all-MiniLM-L6-v2_predicates.pickle"
        },
    "entities_path_map": {
        "umls_clean": "../outputs/umls_nci_bert_embeddings/EntitiesBioLinkBERT-base_UMLS_clean.pickle",
        "umls_noisy": "../outputs/umls_nci_bert_embeddings/EntitiesBertEmbedding_sc_L6-v2_noisy_nci.pickle",
        "umls_mitigated": "../outputs/EntitiesBertEmbedding_augmented_noicy_nci.pickle",
        "CS":"../outputs/umls_nci_bert_embeddings/Entities_tintybert-cs.pickle", #"EntitiesBertEmbedding_sc.pickle",
        "CS_augmented":"../outputs/EntitiesBertEmbedding_sc_augmented.pickle",
        "umls_noisy_mapped": "../outputs/umls_nci_bert_embeddings/EntitiesBertEmbedding_sc_L6-v2_noisy_nci.pickle",
        "gt2kg_mapped_and_old_rel_norm": "outputs/GT2KG_mapped_and_old_rel_norm/sentence-transformers_all-MiniLM-L6-v2_entities.pickle"
        },
    "kg_path_map" : {
    "umls_clean": "../../data/UMLS/clean/KG_NCI_vf.json",
    "umls_noisy": "../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",
    "umls_mitigated": "../../data/UMLS/noisy/augmented/is_a_augmented_MM_mapped_nci_All_R_KG.json",
    "CS": "../../data/Cs/org/original_graph_vf.json",
    "CS_augmented": "../../data/Cs/augmented/augmented_graph_is_rules.json",
    "umls_noisy_mapped": "../../data/UMLS/noisy/relations_mapped/kg_with_mapped_relations_cleaned.json",
    "gt2kg_mapped_and_old_rel_norm": "../../data/UMLS/noisy/org/GT2KG_mapped_and_old_rel_norm.json",
        },
    "GS_path_map" : {
    "umls_clean": "../../data/UMLS/common_nodes.xlsx",
    "umls_noisy": "../../data/UMLS/common_nodes.xlsx",
    "umls_mitigated": "../../data/UMLS/common_nodes.xlsx",
    "umls_noisy_mapped": "../../data/UMLS/common_nodes.xlsx",
    "gt2kg_mapped_and_old_rel_norm": "../../data/UMLS/common_nodes.xlsx",
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

graph_name = os.path.splitext(os.path.basename(config["KG_path"]))[0]
task_name = "_".join(config["training_task"])
run_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{graph_name}_{task_name}").strip("_")

config["wandb_project_name"] = f"Experiments_{run_suffix}"
config["root_save_dir"] = f"ckpt_{run_suffix}"



