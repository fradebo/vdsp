from pyzx.graph.base import BaseGraph, VT, ET
from pyzx import Graph
from pyzx.utils import EdgeType, VertexType
from fractions import Fraction
from itertools import combinations

import networkx as nx

def lcomp(g: BaseGraph[VT,ET], v: VT):
    """assumes we only apply lcomp on X vertices (i.e. Pauli X stabilizer on v has no effect) so only non-adjacent lcomp vertices in minimzation process"""
    vn = list(g.neighbors(v))
    vn.sort()
    for n in vn:
        g.add_to_phase(n,Fraction(1,2))
        # flip edges
        for n2 in vn[vn.index(n)+1:]:
            if g.connected(n,n2):
                g.remove_edge(g.edge(n,n2))
            else:
                g.add_edge(g.edge(n,n2), EdgeType.HADAMARD)

def lcomp_edge_optimize_greedy(g: BaseGraph[VT,ET], vertex_set):
    """greedy approach for minimizing the number of edges in a graph using local complementation"""
    while True:
        vertex_cost_function = [(v,lcomp_cost(g,v)) for v in vertex_set]
        best_vertex, cost = min(vertex_cost_function, key = lambda x: x[1])
        if cost < 0:
            print("apply lcomp",best_vertex,cost)
            lcomp(g, best_vertex)
        else:
            break

def lcomp_cost(g: BaseGraph[VT,ET], v: VT):
    all_edges = set(combinations(g.neighbors(v),2))
    existing_edges = all_edges.intersection(g.edge_set())
    return len(all_edges)-2*len(existing_edges)

def find_maximal_independent_set(g: BaseGraph[VT, ET]):
    """easiest method for finding a maximal independent set; 
    note: this is not an algorithm for finding the maximum(!) independent set which is NP-hard"""
    result = []
    vertex_set = g.vertex_set()
    while vertex_set:
        v = vertex_set.pop()
        result.append(v)
        vertex_set.difference_update(set(g.neighbors(v)))
    return result

def extract_spanning_tree_bfs(g: BaseGraph, start_vertex: VT):
    """maybe better use build_optimal from QTree"""
    tree = g.clone()
    queue = [start_vertex]
    edges = []
    visited = [start_vertex]
    while queue:
        v = queue.pop(0)
        for n in g.neighbors(v):
            if not n in visited:
                visited.append(n)
                queue.insert(0,n)
                edges.append(g.edge(v,n))
    
    all_edges = list(tree.edges())
    for edge in all_edges:
        if not edge in edges:
            tree.remove_edge(edge)
    return tree

def extract_spanning_tree_dfs(g: BaseGraph, start_vertex: VT):
    """maybe better use build_optimal from QTree"""
    tree = g.clone()
    queue = [start_vertex]
    edges = []
    visited = [start_vertex]
    while queue:
        v = queue.pop(0)
        for n in g.neighbors(v):
            if not n in visited:
                visited.append(n)
                queue.append(n)
                edges.append(g.edge(v,n))
    
    all_edges = list(tree.edges())
    for edge in all_edges:
        if not edge in edges:
            tree.remove_edge(edge)
    return tree

if __name__ == "__main__":
    random_graph = nx.erdos_renyi_graph(20,0.3)
    g = Graph()
    vertex_dict = {v: g.add_vertex(VertexType.Z) for v in list(random_graph.nodes) if random_graph.degree(v) > 0}
    for u,v, _ in random_graph.edges.data():
        g.add_edge(g.edge(vertex_dict[u],vertex_dict[v]), EdgeType.HADAMARD)
    
    independent_set = find_maximal_independent_set(g)
    lcomp_edge_optimize_greedy(g, independent_set)
    vertex_degree_function = [(v,len(g.neighbors(v))) for v in g.vertex_set()]
    start_vertex, degree = min(vertex_degree_function, key = lambda x: x[1])
    # start_vertex = list(g.vertices())[0]
    print("choose start vertex",start_vertex, degree)
    tree = extract_spanning_tree_bfs(g, start_vertex)
    print(tree)