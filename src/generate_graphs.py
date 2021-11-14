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
    # We generate a graph with "x" nodes as source, edge types "weight" and
    # "y" nodes as destinations.
    # We then add the weight value of x and y as a feature of the
    # edges of the graph.
    graph_data = {
        ("user_id", "weight", "dst_id"): (data["x"], data["y"])
    }
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    dgl_graph = dgl.graph((x, y))
    dgl_graph.edata["weight"] = torch.tensor(data["weight"].to_numpy())
    return dgl_graph
