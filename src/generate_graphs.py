import pandas as pd
import stellargraph as sg
import dgl
import torch

def generate_stellar_graph(data: pd.DataFrame):
    # We seem to want to build what StellarGraph calls an "heterogeneous graph":
    # More than one node type and/or more than one edge type with and without
    # node/edge features or edge weights, this includes knowledge graphs.
    # Followed this tutorial for building the graph: https://stellargraph.readthedocs.io/en/stable/demos/basics/loading-pandas.html#Multiple-node-types
    users = pd.DataFrame(index=["user-" + str(x) for x in data["user_id"].unique()])
    movies = pd.DataFrame(index=["movie-" + str(x) for x in data["movie_id"].unique()])

    data["user_id"] = data["user_id"].apply(lambda user_id: "user-" + str(user_id))
    data["movie_id"] = data["movie_id"].apply(lambda movie_id: "movie-" + str(movie_id))
    data.rename(columns={
        "user_id": "source",
        "movie_id": "target",
        "score": "weight"
    }, inplace=True)

    return sg.StellarGraph({"user": users, "movie": movies}, data)

def generate_dgl_graph(data: pd.DataFrame):
    # We generate a graph with "user_id" nodes as source, edge types "raiting" and "movie_id" nodes as destinations.
    # Then, add the "raiting" values as weights of edges.
    # + We may add some lines about embeddings of nodes after the embedding vector task is done.

    # Our graph has 2 types of node (user type / movie type), so the graph is heterogeneous
    graph_data = {
        ('user_id', 'raiting', 'movie_id') : (data["user_id"], data["movie_id"]),
        ('movie_id', 'raiting', 'user_id') : (data["user_id"], data["movie_id"])
    }
    dgl_graph = dgl.heterograph(graph_data)
    dgl_graph.edata["raiting"] = torch.tensor(data["raiting"].to_numpy())
    return dgl_graph
