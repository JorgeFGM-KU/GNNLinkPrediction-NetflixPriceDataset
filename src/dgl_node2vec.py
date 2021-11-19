import dgl
def dgl_node2vec(dgl_graph, p, q, walk_length, return_edge_ids) :
    '''
    :param dgl_graph: DGLGraph
    :param p: (float) Likelihood of immediately revisiting a node in the walk
    :param q: (float) Control parameter to interpolate between breadth-first strategy and depth-first strategy
    :param Walk_length: (int) Length of random walks
    :param return_edge_ids: (boolean, default : False) If True, additionally return the edge IDs traversed
    :return traces : (Tensor) A 2-dimensional node ID tensor with shape ``(num_seeds, walk_length + 1)``
            eids : (Tensor, optional) A 2-dimensional edge ID tensor with shape ``(num_seeds, length)``
    '''

    return dgl.sampling.node2vec_random_walk(dgl_graph, dgl_graph.nodes(), p, q, walk_length=walk_length, return_eids=return_edge_ids)

