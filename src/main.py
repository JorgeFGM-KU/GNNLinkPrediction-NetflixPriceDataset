from data_preprocessing import get_dataframe_from_dataset
from generate_graphs import generate_stellar_graph, generate_dgl_graph

# Choose one graph type, do not run bot or it will crash due to changes being
# applied to the "data" Pandas DataFrame hehe

data = get_dataframe_from_dataset(["data/toy_dataset/raw/ratings.txt"])
print(data)

stellar_graph = generate_stellar_graph(data)
# dgl_graph = generate_dgl_graph(data)
print("--- STELLAR GRAPH ---")
print(stellar_graph.info())

#print("--- DGL GRAPH ---")
#print(dgl_graph)