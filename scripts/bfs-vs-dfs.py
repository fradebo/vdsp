import random
import networkx as nx
import json
from scripts.compare_trees import nx_to_vdsp, get_edge_order_bfs, get_edge_order_random, get_edge_order_random_bfs, get_edge_order_random_dfs, get_edge_order_minimal_depth_dfs
from compiler.LOTree import build_optimal_linear, LOTree
import pandas as pd

"""idea: generate multiple bfs and dfs trees for each configuration (number nodes, edge probability)
and then take the average value of fusions"""

def dump_nx_tree(tree, filename):
    tree_map = dict()
    for node in tree.nodes:
        tree_map[node] = list(tree.neighbors(node))
        tree_map[node].sort(key = lambda node: node)

    with open(filename, 'w') as f:
        json.dump(tree_map, f)

def connect_graph(G):
    C = list(nx.connected_components(G))
    for idx in range(1,len(C)):
        G.add_edge(list(C[idx-1])[0],list(C[idx])[0])

def generate_trees(number_nodes, edge_prob, traversal_type, num_samples = 5):
    trees = dict()
    for i in range(num_samples):
        G = nx.erdos_renyi_graph(number_nodes, edge_prob)
        connect_graph(G)
        if traversal_type == "bfs":
            bfs = nx.bfs_tree(G, 0).to_undirected()
            trees["conf_"+str(number_nodes)+'_'+str(edge_prob)+"_type_bfs_sample_"+str(i)] = bfs
        elif traversal_type == "dfs":
            dfs = nx.dfs_tree(G, 0).to_undirected()
            trees["conf_"+str(number_nodes)+'_'+str(edge_prob)+"_type_dfs_sample_"+str(i)] = dfs
    return trees

if __name__ == "__main__":
    random.seed(272)
    all_trees = dict()
    for conf in [(10,0.3),(10,0.5),(8,0.4)]: #,(11,0.1)
        all_trees |= generate_trees(conf[0],conf[1],"bfs")
        all_trees |= generate_trees(conf[0],conf[1],"dfs")

    csvdata = []
    for conf, tree in all_trees.items():
        max_degree = max([val for (node, val) in tree.degree()])
        vdsp_tree = nx_to_vdsp(tree, 0)
        depth = vdsp_tree.depth()
        optimum =  build_optimal_linear(vdsp_tree.head,LOTree(vdsp_tree.head.value), True)
        csvdata.append([max_degree, depth,optimum[1]])
        dump_nx_tree(tree, 'trees/bfs-vs-dfs-trees/'+conf+".json")
        with open("proc_order/"+conf+".json", 'w') as f:
            order = get_edge_order_bfs(optimum[0],True)
            # random.shuffle(order)
            # import pdb
            # pdb.set_trace()
            json.dump({"order": order}, f)
    
    df = pd.DataFrame(csvdata, columns=['max_degree','depth','outer_loops'], index=all_trees.keys())
    df.to_csv('evaluation/bfs-vs-dfs-bfs-11.csv')
        

