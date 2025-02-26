import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", #
    # "out_channels": [256,128],
    "num_layers": 2,
    # "num_bases": 10,
    "alpha": 0.01,
    "max_masking_percentage": 0.3,
    "total_drop_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 512,
    # "cosine_loss_weight": 0.5,
    "shuffle": False,
    "num_neighbors": [500, 500],
    "num_epochs": 120,
    "Entities_path": "../outputs/EntitiesBertEmbeddingAugmented.pickle",
    "Edges_path": "../outputs/PredicatesBertEmbeddingAugmented.pickle",
    "KG_path": "../../data/original_graph_vf.json",
    # "Gs_path": "../../data/gs_vf.xlsx",
    #"Gs_path_no_other": "../../data/original_gs_vf.xlsx",
     "Gs_path_no_other": "../../data/GS_vf.xlsx",
    "core_concepts" : [  'data structure',
                         'cryptography',
                         'software engineering',
                         'computer graphic',
                         'network security',
                         'computer programming',
                         'operating system',
                         'distributed computing',
                         'machine learning'
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
    "training_task" : ["Recons_A"],
    "hyperparams_grid" : {"num_bases": [10], "out_channels": [ [640,512],[512,256],[256,128] ,[128,64], [64,32]]},
    "wandb_project_name": "last_Experiments_GAE",
    "encoders": ["GCN", "RGCN"],
    "decoders": ["GCN","RGCN","MLP"],
    "message_sens": ["source_to_target"],
    "projections": None,
    "root_save_dir": "checkpoints",
}





