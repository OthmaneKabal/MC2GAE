# Entiy Embedding 
import sys
sys.path.append('../../utilities')
import utilities as u
import BertEmbedder as bem
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import argparse

class GraphBERTEmbedder:
    def __init__(self, KG_path, output_path, pretrained_model_name_or_path):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.bertModel = bem.BertEmbedder(self.pretrained_model_name_or_path)
        self.KG = u.read_json_file(KG_path)
        
        ## directory path for the outputs
        self.output_path = output_path
        
        self.Entites_embedding_dict = {} ## result initialisation
        self.Predicates_embedding_dict = {} ## result initialisation
       
    def entities_predicates_Embedding(self):
        for triplet in tqdm(self.KG, desc="Triplets elements Embedding", unit="triple"):
            if triplet["subject"] not in self.Entites_embedding_dict.keys():
                self.Entites_embedding_dict[triplet["subject"]] = self.bertModel.embed_entity(triplet["subject"])
            if triplet["object"] not in self.Entites_embedding_dict.keys():
                self.Entites_embedding_dict[triplet["object"]] = self.bertModel.embed_entity(triplet["object"])
            if triplet["predicate"] not in self.Predicates_embedding_dict.keys():
                self.Predicates_embedding_dict[triplet["predicate"]] = self.bertModel.embed_entity(triplet["predicate"])
                
    def run(self):
        print("Entities and Predicates Embedding using : "+ self.pretrained_model_name_or_path + "...")
        self.entities_predicates_Embedding()
        u.save_to_pickle(self.output_path + "/EntitiesBertEmbeddingAugmented.pickle", self.Entites_embedding_dict)
        u.save_to_pickle(self.output_path + "/PredicatesBertEmbeddingAugmented.pickle", self.Predicates_embedding_dict)
        print("Done")
