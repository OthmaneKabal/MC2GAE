# ## without Edges
# import sys 
# sys.path.append("../../utilities")
# import utilities as u
# import torch
# from torch_geometric.data import Data
# # from torch_geometric.utils import train_test_split_edges
# from torch_geometric.transforms import RandomLinkSplit
# import networkx as nx
# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import to_undirected


# class GraphDataPreparation:
#     def __init__(self, entities_embd_path, kg_path, is_directed=True):
#         self.entities_embd_path = entities_embd_path
#         self.kg_path = kg_path
#         self.built_graph = None
#         self.nxGraph = None
#         self.nodes_index = None
#         self.is_directed = is_directed

#     def decode_indexes(self):
#         inverted_dict = {value: key for key, value in self.nodes_index.items()}
#         return inverted_dict

#     def build_networkx_graph(self):
#         print("Building NetworkX graph")
#         graph_data = u.read_json_file(self.kg_path)
#         embeddings = u.read_pickle_file(self.entities_embd_path)

#         # Create a NetworkX graph
#         if self.is_directed:
#             self.nxGraph = nx.DiGraph()
#         else:
#             self.nxGraph = nx.Graph()

#         # Add nodes with their embeddings
#         for node, emb in embeddings.items():
#             self.nxGraph.add_node(node, emb=emb.clone().detach())

#         # Add edges
#         for i in range(len(graph_data)):
#             self.nxGraph.add_edge(graph_data[i]['subject'], graph_data[i]['object'])

#         # Create node to index mapping
#         self.nodes_index = {node: i for i, node in enumerate(self.nxGraph.nodes())}

#     def build_torch_geometric_data(self):
#         print("Building PyTorch Geometric Data object")
        
#         # Extract node features and edge index
#         node_features = [self.nxGraph.nodes[node]['emb'] for node in self.nxGraph.nodes()]
#         edge_index = [(self.nodes_index[u], self.nodes_index[v]) for u, v in self.nxGraph.edges()]

#         # Convert to PyTorch tensors
#         x = torch.stack(node_features).squeeze(1)
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         if not self.is_directed:
#             edge_index = to_undirected(edge_index)
#         self.built_graph = Data(x=x, edge_index= edge_index)
#         return self.built_graph
        
#     def prepare_graph(self):
#         self.build_networkx_graph()
#         return self.build_torch_geometric_data()
        
#     def split_data(self):
#         transform = RandomLinkSplit(is_undirected= self.is_directed, add_negative_train_samples = False)
#         return transform(self.built_graph)

#     def get_connceted_components(self):
#         connected_components = list(nx.connected_components(self.nxGraph.to_undirected()))
#         return connected_components


#     def get_subgraph_for_component(self, component):
#             subgraph_nodes = list(component)
#             subgraph = self.nxGraph.subgraph(subgraph_nodes)
            
#             # Create node to index mapping for the subgraph
#             subgraph_index = {node: i for i, node in enumerate(subgraph.nodes())}
            
#             # Extract node features and edge index
#             node_features = [subgraph.nodes[node]['emb'] for node in subgraph.nodes()]
#             edge_index = [(subgraph_index[u], subgraph_index[v]) for u, v in subgraph.edges()]
            
#             # Convert to PyTorch tensors
#             x = torch.stack(node_features).squeeze(1)
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#             if not self.is_directed:
#                 edge_index = to_undirected(edge_index)
            
#             subgraph_data = Data(x=x, edge_index=edge_index)
#             return subgraph_data
#############################################################################################
## with Edges

## with edges
import sys 
sys.path.append("../../utilities")
import utilities as u
import torch
from torch_geometric.data import Data
# from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

class GraphDataPreparation:
    def __init__(self, entities_embd_path, kg_path, edges_embd_path=None, is_directed=True):
        self.entities_embd_path = entities_embd_path
        self.edges_embd_path = edges_embd_path
        self.kg_path = kg_path
        self.built_graph = None
        self.nxGraph = None
        self.nodes_index = None
        self.is_directed = is_directed
        self.predicate_to_id = {}  # Dictionary to store predicate type mappings

    def decode_indexes(self):
        inverted_dict = {value: key for key, value in self.nodes_index.items()}
        return inverted_dict

    def build_networkx_graph(self):
        print("Building NetworkX graph")
        graph_data = u.read_json_file(self.kg_path)
        embeddings = u.read_pickle_file(self.entities_embd_path)
        edge_embeddings = u.read_pickle_file(self.edges_embd_path) if self.edges_embd_path else None

        # Create a NetworkX graph
        if self.is_directed:
            self.nxGraph = nx.DiGraph()
        else:
            self.nxGraph = nx.Graph()

        # Add nodes with their embeddings
        for node, emb in embeddings.items():
            self.nxGraph.add_node(node, emb=emb.clone().detach())

        # Add edges with or without embeddings
        for i in range(len(graph_data)):
            subject = graph_data[i]['subject']
            obj = graph_data[i]['object']
            pred = graph_data[i]['predicate']
            edge_data = edge_embeddings[pred] if edge_embeddings is not None else None
            self.nxGraph.add_edge(subject, obj, emb=edge_data)

        # Create node to index mapping
        self.nodes_index = {node: i for i, node in enumerate(self.nxGraph.nodes())}

    def build_torch_geometric_data(self):
        print("Building PyTorch Geometric Data object")
        
        # Extract node features
        node_features = [self.nxGraph.nodes[node]['emb'] for node in self.nxGraph.nodes()]
        
        # Extract edge index and edge attributes (if available)
        edge_index = [(self.nodes_index[u], self.nodes_index[v]) for u, v in self.nxGraph.edges()]
        edge_attr = [self.nxGraph.edges[u, v]['emb'] for u, v in self.nxGraph.edges()] if self.edges_embd_path else None

        # Convert to PyTorch tensors
        x = torch.stack(node_features).squeeze(1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Handle edge attributes if they exist
        edge_attr_tensor = torch.stack(edge_attr).squeeze(1) if edge_attr is not None else None
        
        if not self.is_directed:
            edge_index = to_undirected(edge_index)

        # Include edge attributes in the Data object if they exist
        self.built_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)
        return self.built_graph
        
    def prepare_graph(self):
        self.build_networkx_graph()
        return self.build_torch_geometric_data()
        
    def split_data(self):
        transform = RandomLinkSplit(is_undirected=not self.is_directed, add_negative_train_samples=False)
        return transform(self.built_graph)

    def get_connected_components(self):
        connected_components = list(nx.connected_components(self.nxGraph.to_undirected()))
        return connected_components

    def get_subgraph_for_component(self, component):
        subgraph_nodes = list(component)
        subgraph = self.nxGraph.subgraph(subgraph_nodes)
        
        # Create node to index mapping for the subgraph
        subgraph_index = {node: i for i, node in enumerate(subgraph.nodes())}
        
        # Extract node features and edge index
        node_features = [subgraph.nodes[node]['emb'] for node in subgraph.nodes()]
        edge_index = [(subgraph_index[u], subgraph_index[v]) for u, v in subgraph.edges()]
        edge_attr = [subgraph.edges[u, v]['emb'] for u, v in subgraph.edges()] if self.edges_embd_path else None

        # Convert to PyTorch tensors
        x = torch.stack(node_features).squeeze(1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        edge_attr_tensor = torch.stack(edge_attr).squeeze(1) if edge_attr is not None else None
        
        if not self.is_directed:
            edge_index = to_undirected(edge_index)
        
        # Include edge attributes in the Data object if they exist
        subgraph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)
        
        return subgraph_data

    #####with relation type #################################
    
    # def build_networkx_graph_type(self):
    #     print("Building NetworkX graph")
    #     graph_data = u.read_json_file(self.kg_path)
    #     embeddings = u.read_pickle_file(self.entities_embd_path)
    #     edge_embeddings = u.read_pickle_file(self.edges_embd_path) if self.edges_embd_path else None
    
    #     # Create a NetworkX graph
    #     if self.is_directed:
    #         self.nxGraph = nx.DiGraph()
    #     else:
    #         self.nxGraph = nx.Graph()
    
    #     # Add nodes with their embeddings
    #     for node, emb in embeddings.items():
    #         self.nxGraph.add_node(node, emb=emb.clone().detach())
    
    #     # Add edges with or without embeddings
    #     for i in range(len(graph_data)):
    #         subject = graph_data[i]['subject']
    #         obj = graph_data[i]['object']
    #         pred = graph_data[i]['predicate']
            
    #         # Determine the type based on the predicate
    #         edge_type = 1 if pred == "is" else 0
    #         # Add edge data
    #         edge_data = edge_embeddings[pred] if edge_embeddings is not None else None
    #         self.nxGraph.add_edge(subject, obj, emb=edge_data, type=edge_type)

    #     # Create node to index mapping
    #     self.nodes_index = {node: i for i, node in enumerate(self.nxGraph.nodes())}
    
    
    # def build_torch_geometric_data_with_types(self):
    #     print("Building PyTorch Geometric Data object with relation types")
        
    #     # Extract node features
    #     node_features = [self.nxGraph.nodes[node]['emb'] for node in self.nxGraph.nodes()]
        
    #     # Extract edge index, edge attributes (if available), and edge types
    #     edge_index = [(self.nodes_index[u], self.nodes_index[v]) for u, v in self.nxGraph.edges()]
    #     edge_attr = [self.nxGraph.edges[u, v]['emb'] for u, v in self.nxGraph.edges()] if self.edges_embd_path else None
    #     edge_type = [self.nxGraph.edges[u, v]['type'] for u, v in self.nxGraph.edges()]

    #     # Convert to PyTorch tensors
    #     x = torch.stack(node_features).squeeze(1)
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    #     # Handle edge attributes if they exist
    #     edge_attr_tensor = torch.stack(edge_attr).squeeze(1) if edge_attr is not None else None
        
    #     # Convert edge types to a tensor
    #     edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        
    #     if not self.is_directed:
    #         edge_index = to_undirected(edge_index)

    #     # Include edge attributes and types in the Data object
    #     self.built_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor, edge_type=edge_type_tensor)
    #     return self.built_graph
    

    # def prepare_graph_with_type(self):
    #     self.build_networkx_graph_type()
    #     return self.build_torch_geometric_data_with_types()
    def build_networkx_graph_type(self):
        print("Building NetworkX graph with unique relation type IDs")
        graph_data = u.read_json_file(self.kg_path)[:10000]
        embeddings = u.read_pickle_file(self.entities_embd_path)
        edge_embeddings = u.read_pickle_file(self.edges_embd_path) if self.edges_embd_path else None

        # Initialize NetworkX graph based on directed flag
        self.nxGraph = nx.DiGraph() if self.is_directed else nx.Graph()

        unique_subjects = set(s["subject"] for s in graph_data)
        unique_objects = set(s["object"] for s in graph_data)

        # Add nodes with embeddings
        for node, emb in embeddings.items():
            if (node in unique_subjects or node in unique_objects):
                self.nxGraph.add_node(node, emb=emb.clone().detach())

        # Generate a unique ID for each predicate
        unique_predicates = {entry['predicate'] for entry in graph_data}
        self.predicate_to_id = {predicate: idx for idx, predicate in enumerate(unique_predicates)}

        # Add edges with embeddings and relation type ID
        for entry in graph_data:
            subject = entry['subject']
            obj = entry['object']
            pred = entry['predicate']
            edge_type = self.predicate_to_id[pred]  # Get unique ID for this predicate
            edge_data = edge_embeddings[pred] if edge_embeddings is not None else None
            self.nxGraph.add_edge(subject, obj, emb=edge_data, type=edge_type)

        # Create node to index mapping
        self.nodes_index = {node: i for i, node in enumerate(self.nxGraph.nodes())}

    def build_torch_geometric_data_with_types(self):
        print("Building PyTorch Geometric Data object with unique relation type IDs")
        
        # Extract node features
        node_features = [self.nxGraph.nodes[node]['emb'] for node in self.nxGraph.nodes()]
        
        # Extract edge index, edge attributes, and edge types
        edge_index = [(self.nodes_index[u], self.nodes_index[v]) for u, v in self.nxGraph.edges()]
        edge_attr = [self.nxGraph.edges[u, v]['emb'] for u, v in self.nxGraph.edges()] if self.edges_embd_path else None
        edge_type = [self.nxGraph.edges[u, v]['type'] for u, v in self.nxGraph.edges()]

        # Convert to PyTorch tensors
        x = torch.stack(node_features).squeeze(1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Handle edge attributes if they exist
        edge_attr_tensor = torch.stack(edge_attr).squeeze(1) if edge_attr is not None else None
        
        # Convert edge types to a tensor
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        
        if not self.is_directed:
            edge_index = to_undirected(edge_index)

        # Include edge attributes and types in the Data object
        self.built_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor, edge_type=edge_type_tensor)
        return self.built_graph

    def prepare_graph_with_type(self):
        self.build_networkx_graph_type()
        return self.build_torch_geometric_data_with_types()