import torch
import pandas as pd
from pyg_functions import get_weighted_homogenous_graph, get_node2vec_embedding_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device tat will be used: {device}")
'''
EMBEDDING_SIZE = 64
WALK_LENGTH = 20
CONTEXT_SIZE = 10
WALKS_PER_NODE = 10
NUM_NEGATIVE_SAMPLES = 1
P = 1
Q = 1
'''

movie_graph_data = pd.read_csv("data/toy_dataset/raw/movie_graph.csv")

movie_graph = get_weighted_homogenous_graph(movie_graph_data["x"], movie_graph_data["y"], movie_graph_data["weight"], device=device)
movie_embedding_mat = get_node2vec_embedding_matrix(movie_graph, device=device)