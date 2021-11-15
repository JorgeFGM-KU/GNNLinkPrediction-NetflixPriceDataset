from data_preprocessing import get_dataframe_from_dataset
from generate_graphs import generate_stellar_graph

import numpy as np
import pandas as pd
import stellargraph as sg
import keras
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification

movies_graph_df = pd.read_csv('movie_graph.csv')
movies_graph_df.rename(columns={'x': 'source', 'y': 'target'}, inplace=True)

movies_sg_graph = sg.StellarGraph(edges=movies_graph_df, node_type_default='movie', edge_type_default='weight')

### Node2Vec hyperparameters
walk_number = 100
walk_length = 5
p = 0.5
q = 0.2
batch_size = 50
epochs = 2
emb_size = 64
### ------------------------

biased_random_walker = BiasedRandomWalk(graph=movies_sg_graph, n=walk_number, length=walk_length, p=p, q=q)
unsupervised_samples = UnsupervisedSampler(movies_sg_graph, nodes=list(movies_sg_graph.nodes()), walker=biased_random_walker)
generator = Node2VecLinkGenerator(movies_sg_graph, batch_size)
node2vec = Node2Vec(emb_size, generator=generator)
x_inp, x_out = node2vec.in_out_tensors()
node_gen = Node2VecNodeGenerator(movies_sg_graph, batch_size).flow(movies_graph_df['source'].unique())
embedding_model = keras.Model(inputs=x_inp[0], outputs=x_out[0])
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

""" data = get_dataframe_from_dataset(["data/toy_dataset/raw/ratings.txt"])
print(data)


stellar_graph = generate_stellar_graph(data)
print("--- STELLAR GRAPH ---")
print(stellar_graph.info()) """
