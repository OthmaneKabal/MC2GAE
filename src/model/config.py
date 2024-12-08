import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", #
    "out_channels": [500,250],
    "num_layers": 2,
    "num_bases": 10,
    "alpha": 0.01,
    "max_masking_percentage": 0.3,
    "total_drop_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 10,
    "cosine_loss_weight": 0.5,
    "shuffle": False,
    "num_neighbors": [200, 50],
    "num_epochs": 150,
    "Entities_path": "../outputs/EntitiesBertEmbeddingAugmented.pickle",
    "Edges_path": "../outputs/PredicatesBertEmbeddingAugmented.pickle",
    "KG_path": "../../data/graph_filtred_gpt_20_tst.json",
    "Gs_path": "../../data/gs_vf.xlsx",
    "Gs_path_no_other": "../../data/gs_vf_no_other.xlsx",
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
    "options": ["X"],
    "convE_config":{
                    'embedding_dim': 32,
                    'input_drop': 0.2,
                    'hidden_drop': 0.3,
                    'feat_drop': 0.2,
                    'embedding_shape1': 32,  # Nouvelle valeur
                    'hidden_size': 27776,    # ## 2048 ----> 123008 ; 768 ----> 43648 ; 256 ----> 11904; 512 --> 27776 ; 128----> 3968
                    'label_smoothing': 0.1,
                    'use_bias': True,
                    },
    "coresp_hidden_sizes":{768 : 43648, 512: 27776, 256: 11904, 128: 3968, 64:27776},
    "training_options" : ["contrastive"],
     "hyperparams_grid" : {"num_bases": [10], "out_channels": [[640,512] , [256, 128], [768,768], [100,50]]},
    "save_dir":"ckpt_exp_",
    "wandb_project_name": "Mcontrastive_tst"
}


