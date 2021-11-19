import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec

def get_weighted_homogenous_graph(src_nodes, dst_nodes, weights, device='cpu'):
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float32).reshape(-1,1)
    graph = Data(edge_index=edge_index, edge_attr=weights).to(device)
    return graph

def get_node2vec_embedding_matrix(
    graph,
    embedding_size=64,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1,
    q=1,
    epochs=100,
    print_losses=False,
    device='cpu'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Node2Vec(
        graph.edge_index,
        embedding_dim=embedding_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        sparse=False
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

    for epoch in range(epochs):
        loss = train()
        if print_losses: print(f"Epoch {epoch+1} loss: {loss}")

    return model()