from graphtheory.tree_decompose_bouchet import tree_decompose, complement_neighbors
import networkx as nx
from itertools import combinations

"""Interestingly one can also see that local complementation is in general non-commutative due to the Cliffords applied on the single qubits.
R_x(pi/2) vs. R_z(pi/2). If we have non commuting local complementations, then there are also X and Zs on a qubit, 
if they commute then only Zs or only Xs so also the local operations commute (trivial maybe, 
but we could determine whether local complementations commute or not by just pushing Pauli strings with the phi function?)"""

def edge_optimize(G: nx.Graph):
    G, comp = tree_decompose(G)
    if not comp:
        G, comp = lcomp_edge_optimize_greedy(G, G.nodes)
    return G, comp


def lcomp_edge_optimize_greedy(G: nx.Graph, vertex_set):
    """greedy approach for minimizing the number of edges in a graph using local complementation"""
    lcomps = []
    while True:
        vertex_cost_function = [(v,lcomp_cost(G,v)) for v in vertex_set]
        print(vertex_cost_function)
        best_vertex, cost = min(vertex_cost_function, key = lambda x: x[1])
        if cost < 0:
            # print("apply lcomp",best_vertex,cost)
            complement_neighbors(G, list(G.neighbors(best_vertex)))
            lcomps.append(best_vertex)
        else:
            break
    return G, lcomps

def lcomp_cost(G: nx.Graph, v: int):
    all_edges = set([(c1,c2) if c1 < c2 else (c2,c1) for (c1,c2) in combinations(G.neighbors(v),2)])
    graph_edges = set([(c1,c2) if c1 < c2 else (c2,c1) for (c1,c2) in G.edges])
    existing_edges = all_edges.intersection(graph_edges)
    return len(all_edges)-2*len(existing_edges)