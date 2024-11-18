import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "out_channels": [500,250],
    "num_layers": 2,
    "num_bases": 55,
    "alpha": 0.01,
    "max_masking_percentage": 0.3,
    "total_drop_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 256,
    "shuffle": False,
    "num_neighbors": [100, 100],
    "num_epochs": 150,
    "Entities_path": "../outputs/EntitiesBertEmbeddingAugmented.pickle",
    "Edges_path": "../outputs/PredicatesBertEmbeddingAugmented.pickle",
    "KG_path": "../../data/graph_filtred_gpt_20.json",
    "Gs_path": "../../data/gs_vf.xlsx",
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
    "options": ["X"]
}


