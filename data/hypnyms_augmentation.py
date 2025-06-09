import spacy
from tqdm import tqdm
import sys
sys.path.append('../utilities')
import utilities as u
import itertools
nlp = spacy.load("en_core_web_sm")
from multiprocessing import Pool, cpu_count
## Is-a relationships without "of"
def extract_is_a_relation_basic(term, triplets = None, term_source = None):
    if term_source is None:
        term_source = term
    if triplets is None:
        triplets = []
    doc = nlp(term)
    if len(term.strip().split(' '))>1:
        root = [token for token in doc if token.head == token][0]
#         print(root)
#         tokens = [token.text for token in doc]
        tokens = [token for token in term.strip().split(' ') if token]
        if root.text in tokens:
            root_index = tokens.index(root.text) 
        elif [token for token in tokens if token.startswith(root.text+"-")]:
            root = [token for token in tokens if token.startswith(root.text+"-")][-1]
            root_index = tokens.index(root)
        elif root.text == "-":
            root = [token for token in tokens if root.text in token][-1]
            root_index = tokens.index(root)
        else:
            root_list = [token for token in tokens if root.text in token] 
            if root_list:
                root = root_list[-1]
                root_index = tokens.index(root)
        object_ = ' '.join(tokens[1:root_index+1])
        subject = term
        if subject and object_:
            triplets.append({"source_term":term_source.strip(),"subject":subject.strip(), "predicate": "is-a", "object":object_.strip()})
            return extract_is_a_relation_basic(object_,triplets, term_source)
        else:
            return triplets
    else:
        return triplets
    
    
## Is-a relatioships with of    
def extract_is_a_relation_of(term):
    term_split_of = [elt.strip() for elt in term.split('of') if elt]
    if len(term_split_of) > 1:
        triplets = []
        doc = nlp(term)
        root = [token for token in doc if token.head == token][0].text
        if root == term_split_of[0]:
            for index,value in enumerate(term_split_of[1:]):
                subject = root + " of " + value
                object_ = root
                triplets.append({"source_term":term.strip(),"subject":subject.strip(), "predicate": "is-a", "object":object_.strip()})
                root = subject
                object_ = subject
                
            return triplets
        else:
            return []
    else:
        return []
    
def extract_is_a_relationships(term):
    nlp = spacy.load("en_core_web_sm")
    if "of" in term.split():
        return extract_is_a_relation_of(term)
    else: 
        return extract_is_a_relation_basic(term)
    
## without multiprocessing
def extract_is_a_relationships_all_terms(terms):
    is_a_triplets = []
    for term in tqdm(terms, unit="term"):
        try:
            triplets = extract_is_a_relationships(term)
            is_a_triplets.append(triplets)
        except Exception as e:
            print(f"Erreur lors de l'extraction des relations pour le terme '{term}': {e}")
    flattened_triplets = list(itertools.chain.from_iterable(is_a_triplets))
    return flattened_triplets

# Fonction pour l'extraction des relations "is-a" pour un ensemble de termes en utilisant multiprocessing
def extract_is_a_relationships_all_terms(terms):
    with Pool(cpu_count()) as pool:
        is_a_triplets = list(tqdm(pool.imap(extract_is_a_relationships, terms), total=len(terms), unit="term"))
    flattened_triplets = list(itertools.chain.from_iterable(is_a_triplets))
    return flattened_triplets


## return unique subject and object
def extract_terms_from_graph(graph_path):
    graph = u.read_json_file(graph_path)
    terms = []
    for triplet in graph:
        terms.append(triplet["subject"])
        terms.append(triplet["object"])
    return list(set(terms))
if __name__ == "__main__":
    graph_path = "C:/Users/admin-user/Documents/GitHub/MC2GAE/data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json"
    output_path = "C:/Users/admin-user/Documents/GitHub/MC2GAE/data/UMLS/noisy/augmented/is_a_augmentation_MM_mapped_nci_All_R_KG.json"
    terms = extract_terms_from_graph(graph_path)
    is_a_triplets = extract_is_a_relationships_all_terms(terms)
    u.save_to_json(output_path,is_a_triplets)
    print("Done !")