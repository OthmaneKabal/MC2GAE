import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", #
    # "out_channels": [256,128],
    "seed":42,
    "num_layers": 2,
    # "num_bases": 10,
    "alpha": 0.01,
    "max_masking_percentage": 0.3,
    "total_drop_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 512,
    "test_batch_size": 512,
    # "cosine_loss_weight": 0.5,
    "shuffle": False,
    "num_neighbors": [200,200],
    "num_epochs": 100,

    "Entities_path": "../outputs/EntitiesBertEmbedding_NCI.pickle", #EntitiesBertEmbedding_NCI.pickle",../outputs/EntitiesBertEmbedding_noicy_nci
    "Edges_path": "../outputs/PredicatesBertEmbedding_NCI.pickle",#PredicatesBertEmbedding_NCI.pickle",PredicatesBertEmbedding_noicy_nci

    "Entities_path_cs": "../outputs/EntitiesBertEmbeddingAugmented_cs__.pickle",
    "Edges_path_cs": "../outputs/PredicatesBertEmbeddingAugmented_cs__.pickle",
    "KG_path": "../../data/KG_NCI_vf.json",   #KG_NCI_vf.json",MM_mapped_nci_KG
    "Gs_path_no_other": "../../data/MM_mapped_nci_GS.xlsx",#nci_mm_GS_vf
     "KG_path_cs": "../../data/augmented_graph/augmented_graph_is_rules.json",


    ####
    "Gs_path_no_other_cs": "../../data/GS_vf.xlsx",
    # "Gs_path_no_other": "../../data/GS_communs_terms_nci_mm_test.xlsx",
    ####
    #
    "core_concepts_cs" : [  'data structure',
                         'cryptography',
                         'software engineering',
                         'computer graphic',
                         'network security',
                         'computer programming',
                         'operating system',
                         'distributed computing',
                         'machine learning'
                        ],

    # "core_concepts" : ['Anatomical Structure',
    #      'Finding',
    #      'Group',
    #      'Idea or Concept',
    #      'Intellectual Product',
    #      'Manufactured Object',
    #      'Natural Phenomenon or Process',
    #      'Occupation or Discipline',
    #      'Occupational Activity',
    #      'Organism',
    #      'Organization',
    #      'Substance'],
# "core_concepts" : [
#      'Bacterium',
#      'Biologic Function',
#      'Biomedical Occupation or Discipline',
#      'Body Substance',
#      'Body System',
#      'Chemical',
#      'Clinical Attribute',
#      'Eukaryote',
#      'Food',
#      'Health Care Activity',
#      'Medical Device',
#      'Population Group',
#      'Professional or Occupational Group',
#      'Research Activity',
#      'Spatial Concept',
#      'Virus'
# ],
    "core_concepts" : [
    "Body Part, Organ, or Organ Component",
    # "Cell",
    "Disease or Syndrome",
    "Finding",
    # "Gene or Genome",
    # "Geographic Area",
    "Intellectual Product",
    "Laboratory Procedure",
    "Organic Chemical",
    "Pharmacologic Substance",
    "Therapeutic or Preventive Procedure"
],

    "convE_config":{
                    'embedding_dim': 512,
                    'input_drop': 0.2,
                    'hidden_drop': 0.3,
                    'feat_drop': 0.2,
                    'embedding_shape1': 32,  # Nouvelle valeur
                    'hidden_size': 27776,    # ## 2048 ----> 123008 ; 768 ----> 43648 ; 256 ----> 11904; 512 --> 27776 ; 128----> 3968
                    'label_smoothing': 0.1,
                    'use_bias': True,
                    },
    "coresp_hidden_sizes":{768 : 43648, 512: 27776, 256: 11904, 128: 3968, 64:27776},
    "training_task" : ["Recons_X"],
    "hyperparams_grid" : {"num_bases": [10], "out_channels": [ [384,256],[256,128]]},
    "wandb_project_name": "last_Experiments_Recons_X_09_05",
    "encoders": ["GCN"],
    "decoders": ["MLP","RGCN"],

    "message_sens": ["source_to_target"],
    "projections": None,
    "root_save_dir": "checkpoints",
    "param_combinations": [{"encoder": "GCN","decoder":"GCN","out_channels":[640,512]},
                           {"encoder": "RGCN","decoder":"RGCN","out_channels":[640,512]},
                           {"encoder": "RGCN","decoder":"MLP","out_channels":[640,512]}
                           ]
}




