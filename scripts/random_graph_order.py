import networkx as nx
import numpy as np
import sys
sys.path.append('..')
import random
random.seed(1234)
import math

from firstpassage.FullProgram_3GHZ import get_first_passage_times
from graphtheory.approximate_min_deg_st import approximate_min_deg_st

def connect_graph(G):
    C = list(nx.connected_components(G))
    for idx in range(1,len(C)):
        G.add_edge(list(C[idx-1])[0],list(C[idx])[0])

def generate_graphs(num_vertices, edge_prob, num_samples = 5):
    graphs = []
    for _ in range(num_samples):
        G = nx.erdos_renyi_graph(num_vertices, edge_prob)
        G = G.to_undirected()
        connect_graph(G)
        graphs.append(G)
    return graphs

def eval(G, succ_probs, verbose=False):
    edge_set = set([(v1,v2) if v1 < v2 else (v2,v1) for (v1,v2) in G.edges])
    st = get_spanning_tree(G, traverse_method='bfs',depth='max', min_degree=False)
    st_proc_order = get_spanning_tree_order(st,traverse_method='largest_first',start='largest')
    st_edge_set = set(st_proc_order)
    proc_order_tree_first = st_proc_order + list(edge_set.difference(st_edge_set))
    proc_order_random = proc_order_tree_first[:] #list(edge_set) #list(edge_set.difference(st_edge_set)) + list(st_edge_set)
    random.shuffle(proc_order_random)

    print("Graph: ",G.edges)
    print("spanning tree:",st.edges)
    print("tree proc order:",proc_order_tree_first)
    print("random proc order:",proc_order_random)

    return (get_first_passage_times(proc_order_tree_first, succ_probs),get_first_passage_times(proc_order_random, succ_probs, verbose=verbose))

def average_samples(fp_times, succ_probs):
    average_tree_improve = dict()
    for p in succ_probs:
        average_tree_improve[p] = 0
        for i in range(len(fp_times)):
            average_tree_improve[p] += fp_times[i][0][p]/fp_times[i][1][p]
        average_tree_improve[p] = 1-(average_tree_improve[p]/len(fp_times))
    return average_tree_improve


def minmax_depth_dfs(G, dfs=True, min=False):
    """toggle dfs=True: dfs, else bfs; toggle min=True: min depth, else max depth"""
    longest_tree = (None,1000) if min else (None,0)
    for v in list(G.nodes):
        dfs_tree = nx.dfs_tree(G, v) if dfs else nx.bfs_tree(G, v)
        depth = len(nx.dag_longest_path(dfs_tree))
        if min and depth < longest_tree[1] or not min and depth > longest_tree[1]:
            longest_tree = (dfs_tree,depth)
        
    return longest_tree[0].to_undirected()


"""3 Parameter: traverse_method(bfs/dfs), depth(min/random/max), min_degree(True/False)"""
def get_spanning_tree(G, traverse_method='dfs', depth='random', min_degree=False):
    if depth == 'random':
        if traverse_method == 'dfs':
            T = nx.dfs_tree(G, list(G.nodes)[0]).to_undirected()
        else:
            T = nx.bfs_tree(G, list(G.nodes)[0]).to_undirected()
    else:
        T = minmax_depth_dfs(G, traverse_method == 'dfs', depth == 'min')
    if min_degree:
        T = approximate_min_deg_st(G,T.to_undirected())
    return T

def get_spanning_tree_order(T, traverse_method='dfs', start='largest'):
    """Given fixed tree, determine fusion order
    possible traversals: dfs, bfs, largest nodes first, smallest nodes first
    how to determine the start node?
    can we somehow incorporate francescos dfs graphs? """
    node_degrees_map = get_node_degrees_map(T)
    
    start_vertex = node_degrees_map[min(node_degrees_map.keys())][0] if start != 'largest' else node_degrees_map[max(node_degrees_map.keys())][0]
    if traverse_method == 'dfs':
        return [(v1,v2) if v1 < v2 else (v2,v1) for (v1,v2) in nx.dfs_edges(T, start_vertex)]
    elif traverse_method == 'bfs':
        return [(v1,v2) if v1 < v2 else (v2,v1) for (v1,v2) in nx.bfs_edges(T, start_vertex)]
    else:
        proc_order = []
        for deg in sorted(node_degrees_map.keys(), reverse= (traverse_method == 'largest_first')):
            print("deg",deg,"verts",node_degrees_map[deg])
            for v in node_degrees_map[deg]:
                for n in T.neighbors(v):
                    edge = (v,n) if v < n else (n,v)
                    if not edge in proc_order:
                        proc_order.append(edge)
        return proc_order

def get_node_degrees_map(G):
    node_degrees_map = dict()
    for v in G.nodes:
        n = G.degree[v]
        if not n in node_degrees_map:
            node_degrees_map[n] = [v]
        else:
            node_degrees_map[n].append(v)
    return node_degrees_map

# def graph_fusion_strategy(G):
#     degree_map = get_node_degrees_map(G)
#     num_partitions = math.floor(G.nodes/3 + 1)



if __name__ == "__main__":

    succ_probs = np.linspace(0.5,1,10)
    graphs = generate_graphs(7, 0.4, 20)
    fp_times = []
    for i,graph in enumerate(graphs):
        res = eval(graph, succ_probs)
        # print(res)
        fp_times.append(res)
    # import pdb
    # pdb.set_trace()
    avg_tree_improve = average_samples(fp_times, succ_probs)
    print("avg tree:",avg_tree_improve)
    import matplotlib.pyplot as plt
    plt.xlabel('Success Probability')
    plt.ylabel('Improvement tree vs. random fusion order')
    plt.title('Avg. FP times percentage for tree vs. random fusion order (n=7)')
    plt.xticks(rotation=45)
    display_dict = dict()
    for k,v in avg_tree_improve.items():
        display_dict[str(round(k, 2))] = v
    plt.plot(*zip(*display_dict.items()))
    plt.tight_layout()
    plt.savefig('tree-vs-random-7.pdf')
    # print("avg random:",avg_random)