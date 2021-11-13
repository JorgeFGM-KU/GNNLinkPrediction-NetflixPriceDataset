from data_preprocessing import get_dataframe_from_dataset
from generate_graphs import generate_stellar_graph


data = get_dataframe_from_dataset(["data/toy_dataset/raw/ratings.txt"])
print(data)

stellar_graph = generate_stellar_graph(data)
print("--- STELLAR GRAPH ---")
print(stellar_graph.info())

