import networkx as nx

import matplotlib.pyplot as plt
from pathlib import Path
"""https://www.cs.cmu.edu/afs/cs/academic/class/15210-s12/www/lectures/lecture18.pdf
Using the contracted graph, we can find a mxaimal size of disjoint edges in polynomial time!
https://math.stackexchange.com/questions/911788/maximal-size-set-of-disjoint-edges
"""

def debug_draw(ghz_contraction):
    plt.title("Contracted Graph "+str(len(ghz_contraction.nodes)))
    nx.draw(ghz_contraction,with_labels=True)
    path_name = "Figs of contractions"
    figs_path = Path(path_name)
    figs_path.mkdir(exist_ok=True)
    plt.savefig(path_name+"/"+str(len(ghz_contraction.nodes))+"w.png")
    plt.close()
    print("Figure saved to \""+path_name+"/"+str(len(ghz_contraction.nodes))+"w.png\"")

def approximate_minimal_fpt_fusion_order(fusion_order):
    ghz_contraction = get_ghz_contracted_graph(fusion_order)
    vertex_map = {i: set([i*3,i*3+1,i*3+2]) for i in ghz_contraction.nodes}
    # debug_draw(ghz_contraction)
    result = []
    while len(ghz_contraction.nodes) > 1:
        for (v1,v2) in nx.max_weight_matching(ghz_contraction):
            v1,v2 = (v1,v2) if v1 < v2 else (v2,v1)
            for (q1,q2) in fusion_order:
                q1,q2 = (q1,q2) if q1 < q2 else (q2,q1)
                if (q1 in vertex_map[v1] and q2 in vertex_map[v2]) or (q2 in vertex_map[v1] and q1 in vertex_map[v2]):
                    result.append((q1,q2))
                    nx.contracted_nodes(ghz_contraction, v1,v2, self_loops=False, copy=False)
                    vertex_map[v1].update(vertex_map[v2])
                    del vertex_map[v2]
                    # print("fuse ",q1,q2,"ghz_contraction for",v1,v2,"new vertex map",vertex_map)
                    break
            # debug_draw(ghz_contraction)
            # print(ghz_contraction.edges)
        
            #here we need to add all other fusions which are not joining any disjoint subgraphs
            for (q1,q2) in fusion_order:
                q1,q2 = (q1,q2) if q1 < q2 else (q2,q1)
                if not (q1,q2) in result:
                    for qubit_group in vertex_map.values():
                        if q1 in qubit_group and q2 in qubit_group:
                            result.append((q1,q2))
        # print("temp result",result)
            
    return result

def get_ghz_contracted_graph(fusion_order):
    edges = []
    for (q1,q2) in fusion_order:
        node1 = int(q1/3)
        node2 = int(q2/3)
        edges.append((node1,node2))
    return nx.Graph(edges)

#TODO: add random order (but we don't really need a method for this, just random shuffle) and tree order? 
# Tree order is kind of just what the spanning tree generated fusion network already gives us? Yes, maybe this is sufficient. 
# We don't have tree orders for ghz_mapping and random_network then, but anyway tree order should be worse than random_order and approx_min_order

# def adjust_qubit_order(fusion_order, num_qubits):
#     """normalizes fusion order so that the first fusion also happens between modes 0-2 and 3-5"""
#     subtraction = int(fusion_order[0][0]/3)*3
#     new_fusion_order = []
#     for fusion in fusion_order:
#         new_fusion = []
#         for qubit in fusion:
#             if qubit < subtraction:
#                 new_fusion.append(num_qubits-subtraction+qubit)
#             else:
#                 new_fusion.append(qubit-subtraction)
#         new_fusion_order.append(tuple(sorted(new_fusion)))
#     return new_fusion_order

def adjust_qubit_order(fusion_order):
    """normalizes fusion order so that earlier fusions also happen on earlier modes if possible"""
    next_free_mode = 0
    new_fusion_order = []
    vertex_pos_map = {}
    for fusion in fusion_order:
        for qubit in fusion:
            if not qubit in vertex_pos_map:
                start = int(qubit/3)*3
                for j in range(start,start+3):
                    vertex_pos_map[j] = next_free_mode
                    next_free_mode += 1
        new_fusion_order.append(tuple(sorted([vertex_pos_map[fusion[0]], vertex_pos_map[fusion[1]]])))
    return new_fusion_order