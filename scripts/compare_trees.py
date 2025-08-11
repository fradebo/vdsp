from compiler.Graph import create_tree_dfs, Tree, TreeNode
from compiler.LOTree import LOTree, build_optimal_linear
from graphtheory.approximate_min_deg_st import approximate_min_deg_st
import networkx as nx
import random
import json
import perceval as pcvl
import matplotlib.pyplot as plt
import pandas as pd

## helper functions; TODO: fix graph structure, there should not be nx + pyzx graphs + Trees but just one format
def vdsp_to_nx(T):
    edge_list = []
    current = [T.head]
    while current:
        next = []
        for node in current:
            for child in node.children:
                edge_list.append((node.value,child.value))
                next.append(child)
        current = next
    return nx.Graph(edge_list)

def nx_to_vdsp(T: nx.Graph, head):
    head_node = TreeNode(head)
    vdsp_tree = Tree(head_node)
    current = [head_node.value]
    visited = [head_node.value]
    while current:
        next = []
        for node in current:
            for neighbor in T.neighbors(node):
                if not neighbor in visited:
                    visited.append(neighbor)
                    vdsp_tree.add_edge(node,neighbor)
                    next.append(neighbor)
        current = next
    return vdsp_tree

def min_degree_dfs(G: nx.Graph):
    T = dfs(G)
    return approximate_min_deg_st(G,T)

def dfs(G: nx.Graph):
    v = list(G.nodes)[0]
    return vdsp_to_nx(create_tree_dfs(G,v))

def longest_line_dfs(G: nx.Graph):
    longest_tree = (None,0)
    for v in list(G.nodes):
        T = create_tree_dfs(G,v)
        depth = T.depth()
        if depth > longest_tree[1]:
            longest_tree = (T,depth)
    return vdsp_to_nx(longest_tree[0])

def min_degree_longest_line(G: nx.Graph):
    T = longest_line_dfs(G)
    return approximate_min_deg_st(G,T)

def get_optimal_tree(g: nx.Graph, st_extraction_method = dfs):
    optimal_tree = None
    optimal_outer_loops = -1
    t = st_extraction_method(g)
    for head in t.nodes:
        vdsp_tree = nx_to_vdsp(t, head)
        q = LOTree(head)
        # print("before build optimal")
        current_optimal_tree, current_loops = build_optimal_linear(vdsp_tree.head,q)
        #    current_loops = build_optimal_linear(vdsp_tree.head,q, fusion_method="type2")

        if optimal_outer_loops == -1 or current_loops < optimal_outer_loops:
            optimal_outer_loops = current_loops
            optimal_tree = current_optimal_tree
    return (optimal_tree, optimal_outer_loops)

def dump_tree_and_proc_order(tree: LOTree, tree_filename, proc_order_filename):
    proc_order = {'order': get_edge_order2(tree)}
    tree_map = dict()
    # for i,node in enumerate(tree.fusion_order()):
    #     proc_order[node] = i
    # todo: traverse tree in bfs order and set the correct order from fusion order
    
    
    # convert tree indexing so that root is element 0
    conv_map = dict({tree.tree.head.value: 0})
    queue = [tree.tree.head]
    counter = 1
    while queue:
        current_node = queue.pop()
        for child in reversed(current_node.children):
            conv_map[child.value] = counter
            counter += 1
            queue.append(child)
    # import pdb
    # pdb.set_trace()
    for node in tree.tree.vertices:
        tree_map[conv_map[node.value]] = [conv_map[neighbor.value] for neighbor in [node.parent]+node.children if neighbor]
    
    # for node in tree.tree.vertices:
    #     for child in node.children:
    #         tree_map[child.value].append(node.value)
    
    with open(tree_filename, 'w') as f:
        json.dump(tree_map, f)
    
    with open(proc_order_filename, 'w') as f:
        json.dump(proc_order, f)

def get_edge_order_minimal_depth_dfs(tree: LOTree, reversed=False):
    order = []
    conv_map = get_edge_order_dict(tree)
    visited = [tree.tree.head.value]
    stack = [tree.tree.head]
    while stack:
        node = stack.pop()
        if not reversed:
            # specific to dfs search we need to inverse the reversing process...
            node.children.reverse()
        for child in node.children:
            if not child.value in visited:
                visited.append(child.value)
                stack.append(child)

        if node.parent:
            order.append(conv_map[frozenset([node.parent.value,node.value])])
    return order

def get_edge_order_bfs(tree: LOTree, reversed=False):
    conv_map = get_edge_order_dict(tree)
    order = []
    visited = [tree.tree.head.value]
    stack = [tree.tree.head]
    while stack:
        node = stack.pop(0)
        if reversed:
            node.children.reverse()
        for child in node.children:
            if not child.value in visited:
                visited.append(child.value)
                stack.append(child)

        if node.parent:
            order.append(conv_map[frozenset([node.parent.value,node.value])])
    return order

def get_edge_order_random_bfs(tree: LOTree):
    conv_map = get_edge_order_dict(tree)
    order = []
    visited = [tree.tree.head.value]
    stack = [tree.tree.head]
    while stack:
        node = stack.pop(0)
        children = node.children
        random.shuffle(children)
        for child in children:
            if not child.value in visited:
                visited.append(child.value)
                stack.append(child)

        if node.parent:
            order.append(conv_map[frozenset([node.parent.value,node.value])])
    return order

def get_edge_order_random(tree: LOTree):
    conv_map = get_edge_order_dict(tree)
    order = []
    visited = [tree.tree.head.value]
    stack = [tree.tree.head]
    while stack:
        node = stack.pop()
        for child in node.children:
            if not child.value in visited:
                visited.append(child.value)
                stack.append(child)

        if node.parent:
            order.append(conv_map[frozenset([node.parent.value,node.value])])
        random.shuffle(stack)
    return order

def get_edge_order_random_dfs(tree: LOTree):
    conv_map = get_edge_order_dict(tree)
    order = []
    visited = [tree.tree.head.value]
    stack = [tree.tree.head]
    while stack:
        node = stack.pop()
        children = node.children
        random.shuffle(children)
        for child in children:
            if not child.value in visited:
                visited.append(child.value)
                stack.append(child)

        if node.parent:
            order.append(conv_map[frozenset([node.parent.value,node.value])])
    return order


def get_edge_order_dict(tree: Tree):
    edge_idx = 0
    visited = []
    conv_map = {}
    tree.tree.vertices.sort(key= lambda node: node.value)
    for parent in tree.tree.vertices:
        neighbor_list = [ n for n in parent.children + [parent.parent] if n]
        neighbor_list.sort(key=lambda node: node.value)
        for neighbor in neighbor_list:
            if not neighbor.value in visited:
                conv_map[frozenset([parent.value,neighbor.value])] = edge_idx
                edge_idx += 1
        visited.append(parent.value)

    return conv_map

def print_tree(lotree: LOTree):
    expand = [(lotree.tree.head,0)]
    while expand:
        vertex,depth = expand.pop()
        print(" "*(4*(depth-1))+("|---" if depth > 0 else "")+str(vertex.value))
        for child in reversed(vertex.children):
            expand.append((child, depth+1))

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
    # import pdb
    # pdb.set_trace()
    with open(filename, 'w') as f:
        json.dump(tree_dict, f)


def compare_methods(G: nx.Graph):
    methods = {'min_degree': min_degree_dfs, 'dfs': dfs, 'longest_line': longest_line_dfs, 'min_degree_longest_line': min_degree_longest_line}
    data = []
    for name, method in methods.items():
        # tree = method(G)
        # dump_tree2(nx_to_vdsp(tree, 0),'trees/tree_'+name+'.json')
        optimum = get_optimal_tree(G, method)
        dump_tree_and_proc_order(optimum[0],'trees/tree_'+name+'.json','proc_order/tree_'+name+'.json')
        print_tree(optimum[0])
        max_degree = max([len(node.children)+1 for node in optimum[0].tree.vertices])
        depth = optimum[0].tree.depth()
        data.append([max_degree, depth,optimum[1]])
        # new_data = pd.DataFrame([[max_degree, depth,optimum[1]]], columns=['max_degree','depth','outer_loops'], index=[name])
        # df = pd.concat([df,new_data])
        print("num outer loops ",name,":",optimum[1], "min degree", max([len(node.children)+1 for node in optimum[0].tree.vertices]))
        # pcvl.pdisplay(optimum[0].circuit, output_format='SVG')
    df = pd.DataFrame(data, columns=['max_degree','depth','outer_loops'], index=methods.keys())
    df.to_csv('First Passage/FP_data.csv')

if __name__ == "__main__":
    random.seed(2334)
    G = nx.erdos_renyi_graph(8,0.3)

    # ensure connectedness
    C = list(nx.connected_components(G))
    for idx in range(1,len(C)):
        G.add_edge(list(C[idx-1])[0],list(C[idx])[0])
    
    if not nx.is_connected(G):
        print("G is not connected, abort")
    else:
        print("G with edges",G.edges)
        nx.draw(G)
        plt.savefig("original_graph.png")

        compare_methods(G)

