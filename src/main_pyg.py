import torch
import pandas as pd
from torch_geometric.nn.models import GraphSAGE
from data_preprocessing import get_dataframe_from_dataset
from pyg_functions import get_weighted_homogenous_graph, get_node2vec_embedding_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device tat will be used: {device}")

N2V_EMBEDDING_SIZE = 64
N2V_WALK_LENGTH = 20
N2V_CONTEXT_SIZE = 10
N2V_WALKS_PER_NODE = 10
N2V_NUM_NEGATIVE_SAMPLES = 1
N2V_P = 1
N2V_Q = 1

GS_HIDDEN_CHANNELS = 32
GS_NUM_LAYERS = 2
GS_OUT_CHANNELS = 6

movie_graph_data = pd.read_csv("data/toy_dataset/raw/movie_graph.csv")
ratings_graph_data = get_dataframe_from_dataset(["data/toy_dataset/raw/ratings.txt"])

movie_graph = get_weighted_homogenous_graph(movie_graph_data["x"], movie_graph_data["y"], movie_graph_data["weight"], device=device)
movie_embedding_mat = get_node2vec_embedding_matrix(
    movie_graph,
    embedding_size=N2V_EMBEDDING_SIZE,
    walk_length=N2V_WALK_LENGTH,
    context_size=N2V_CONTEXT_SIZE,
    walks_per_node=N2V_WALKS_PER_NODE,
    num_negative_samples=N2V_NUM_NEGATIVE_SAMPLES,
    p=N2V_P,
    q=N2V_Q,
    device=device
)

graph_sage_model = GraphSAGE(
    in_channels=N2V_EMBEDDING_SIZE,
    hidden_channels=GS_HIDDEN_CHANNELS,
    num_layers=GS_NUM_LAYERS,
    out_channels=GS_OUT_CHANNELS
).to(device)

loader = graph_sage_model.loader(batch_size=128, shuffle=True, num_workers=0)
optimizer = torch.optim.Adam(list(graph_sage_model.parameters()), lr=0.01)