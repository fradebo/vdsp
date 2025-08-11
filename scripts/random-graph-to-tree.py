from graphtheory.edge_optimize_zx import find_maximal_independent_set, lcomp_edge_optimize_greedy
from compiler.QTree import create_tree_dfs, build_optimal, QTree
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx import Graph
from pyzx.utils import EdgeType, VertexType
import networkx as nx
import random
import json

"""
In this script we generate random erdös renyi graphs and extract dfs spanning trees iterating through all vertices in the graph as root
We analyze the number of outer loops and dump the trees in json format for further processing (i.e.: calculation of first passage time)
"""


def create_random_graph(num_vertices, edge_prob):
    random_graph = nx.erdos_renyi_graph(num_vertices,edge_prob)
    g = Graph()
    vertex_dict = {v: g.add_vertex(VertexType.Z,random.random()*num_vertices/2,random.random()*num_vertices/2) for v in list(random_graph.nodes) if random_graph.degree(v) > 0}
    for u,v, _ in random_graph.edges.data():
        g.add_edge(g.edge(vertex_dict[u],vertex_dict[v]), EdgeType.HADAMARD)
    g.normalize()
    return g

def minimize_edges(g: BaseGraph[VT,ET]):
    independent_set = find_maximal_independent_set(g)
    lcomp_edge_optimize_greedy(g, independent_set)

def dump_tree(tree, filename):
    tree_dict = dict()
    for v in tree.vertices:
        neighbors = v.children + [v.parent]
        tree_dict[v.value] = [neighbor.value for neighbor in neighbors if neighbor]
    print(tree_dict)
    with open(filename, 'w') as f:
        json.dump(tree_dict, f)

def dump_tree2(tree, filename):
    """same as dump_tree but we change the indexing, so that vertex 0 is always root 
    this is later needed for first passage time calculation where 0 is always the vertex to start the fusion from
    yet still not optimal, because in first passage time calculation we cannot yet fix a specific fusion order"""
    conv_map = dict({tree.head.value: 0})
    queue = [tree.head]
    counter = 1
    # convert tree indexing so that root is element 0
    while queue:
        current_node = queue.pop()
        for child in reversed(current_node.children):
            conv_map[child.value] = counter
            counter += 1
            queue.append(child)
    
    tree_dict = dict()
    for v in tree.vertices:
        neighbors = v.children + [v.parent]
        tree_dict[conv_map[v.value]] = [conv_map[neighbor.value] for neighbor in neighbors if neighbor]
    print(tree_dict)
    with open(filename, 'w') as f:
        json.dump(tree_dict, f)
            


def get_optimal_tree(g: BaseGraph[VT,ET]):
    optimal_tree = None
    optimal_outer_loops = -1
    for v in g.vertices():
        t = create_tree_dfs(g, list(g.vertices())[v])
        q = QTree(t.head.value)
        current_loops = build_optimal(t.head,q)
        if optimal_outer_loops == -1 or current_loops < optimal_outer_loops:
            optimal_outer_loops = current_loops
            optimal_tree = t
    return (optimal_tree, optimal_outer_loops)

def get_all_dfs_trees(g: BaseGraph[VT,ET]):
    dfs_trees = []
    for v in g.vertices():
        t = create_tree_dfs(g, list(g.vertices())[v])
        q = QTree(t.head.value)
        current_loops = build_optimal(t.head,q)
        dfs_trees.append((t,current_loops))
    return dfs_trees

if __name__ == "__main__":
    random.seed(1234)
    g = create_random_graph(11,0.4)
    print("original graph with",g.num_edges(),"edges")
    minimize_edges(g)
    print("after edge minimzer",g.num_edges(),"edges")
    for idx, (tree, num_outer_loops) in enumerate(get_all_dfs_trees(g)):
        print("tree",idx,"has max degree of",max(map(lambda x: len(x.children)+1 if x != tree.head else len(x.children), tree.vertices)))
        print("tree",idx,"has number of outer loops",num_outer_loops)
        print("tree head is",tree.head.value)
        # dump_tree(tree, 'trees/tree'+str(idx)+'.json')
        dump_tree2(tree, 'trees/tree'+str(idx)+'.json')