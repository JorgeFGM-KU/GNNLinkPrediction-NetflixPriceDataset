from data_preprocessing import get_dataframe_from_dataset
from generate_graphs import generate_stellar_graph, generate_dgl_graph
from dgl_node2vec import dgl_node2vec
from construct_movie_graph import construct_movie_graph


data = get_dataframe_from_dataset(["data/toy_dataset/raw/ratings.txt"])
movie_graph = construct_movie_graph(data)
print(movie_graph)

#user_graph = construct_user_graph(data)

dgl_graph = generate_dgl_graph(movie_graph)

print("--- DGL GRAPH ---")
print(dgl_graph)
print("----- DGL node2vec -----")
node2vec = dgl_node2vec(dgl_graph, p=1, q=1, Walk_length=6, return_edge_ids=False)
print(node2vec)
