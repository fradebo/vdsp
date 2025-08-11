import networkx as nx
from graphtheory.approximate_min_deg_st import approximate_min_deg_st
import random

"""3 Parameter: traverse_method(bfs/dfs), depth(min/random/max), min_degree(True/False)"""
def get_spanning_tree(G: nx.Graph, traverse_method='dfs', depth='random', min_degree=False):
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

def minmax_depth_dfs(G, dfs=True, min=False):
    """toggle dfs=True: dfs, else bfs; toggle min=True: min depth, else max depth"""
    longest_tree = (None,1000) if min else (None,0)
    for v in list(G.nodes):
        dfs_tree = nx.dfs_tree(G, v) if dfs else nx.bfs_tree(G, v)
        depth = len(nx.dag_longest_path(dfs_tree))
        if min and depth < longest_tree[1] or not min and depth > longest_tree[1]:
            longest_tree = (dfs_tree,depth)
        
    return longest_tree[0].to_undirected()

def build_fusion_network_from_spanning_tree(sp_tree: nx.Graph,head, G: nx.Graph, traversal_mode = 'dfs', verbose=False):
    """given the desired graph and a spanning tree of it, we construct a fusion network as follows:
    spanning tree head is assigned a ghz + two neighbors
    we then traverse the tree in some order (which?) updating the labels and adding complicated edges corresponding to graph"""
    fusions = []
    root_neighbors = list(sp_tree.neighbors(head))
    edges_to_create = set([(v1,v2) if v1 < v2 else (v2,v1) for (v1,v2) in G.edges])
    if len(root_neighbors) < 2:
        mid = root_neighbors[0]
        second = set(sp_tree.neighbors(mid)).difference(set([head])).pop()
        initial_ghz = (head, mid, second)
        vertex_pos_map = {head: 0, mid: 1, second: 2}
    else:
        initial_ghz = (root_neighbors[0],head,root_neighbors[1])
        vertex_pos_map = {root_neighbors[0]: 0,head: 1,root_neighbors[1]: 2}
    next_free_qubit = 3
    edges_to_create.discard((initial_ghz[0],initial_ghz[1] if initial_ghz[0] < initial_ghz[1] else initial_ghz[1],initial_ghz[0]))
    edges_to_create.discard((initial_ghz[1],initial_ghz[2] if initial_ghz[1] < initial_ghz[2] else initial_ghz[2],initial_ghz[1]))
    if verbose: print("initial ghz",initial_ghz)
    # traverse spanning tree in some order
    edge_order = list(nx.bfs_edges(sp_tree,head) if traversal_mode == 'bfs' else nx.dfs_edges(sp_tree,head))
    edge_order = [(v1,v2) if v1 < v2 else (v2,v1) for (v1,v2) in edge_order]
    
    if verbose: print("traversed edge order",edge_order)
    for edge in edge_order:
        parent,child = (edge[0],edge[1]) if edge[0] in vertex_pos_map else (edge[1],edge[0])

        if not child in initial_ghz:
            #append tree edge
            if verbose: print("append tree edge",parent,child)
            fusions.append((vertex_pos_map[parent],next_free_qubit))
            vertex_pos_map[parent] = next_free_qubit + 1
            vertex_pos_map[child] = next_free_qubit + 2
            next_free_qubit += 3
        else:
            if verbose: print("skip tree edge",parent,child)
        edges_to_create.discard(edge)

        for neighbor in G.neighbors(child):
            # if neighbor in initial_ghz and child in initial_ghz:
            #     if abs(initial_ghz.index(neighbor) - initial_ghz.index(child)) == 1: 
            #         #spaghetti check whether graph edge already covered with initial ghz mapping
            #         continue
            sorted_edge = (child,neighbor) if child < neighbor else (neighbor,child)
            if  neighbor in vertex_pos_map and sorted_edge in edges_to_create and not sorted_edge in edge_order:
                #create graph edge
                if verbose: print("append graph edge",child,neighbor)
                fusions.append((vertex_pos_map[child],next_free_qubit))
                fusions.append((next_free_qubit+2,next_free_qubit+3))
                fusions.append((next_free_qubit+5,vertex_pos_map[neighbor]))
                vertex_pos_map[child] = next_free_qubit+1
                vertex_pos_map[neighbor] = next_free_qubit+4
                next_free_qubit += 6
                edges_to_create.discard(sorted_edge)

    return fusions

def build_fusion_network_from_ghz_partitions(partitioning, G: nx.Graph):
    """We first initialize all 3-qubit partitions and only then add difficult edges between them.
    Problem: also non-full partitions need to be initialized with a ghz. Here we add edges to other partitions with the initialization process, 
    because otherwise we would waste qubits"""
    fusion_network = []
    vertex_pos_map = {}
    qubit_count = 0
    remaining_edges = set([(v1,v2) if v1 < v2 else (v2,v1) for v1,v2 in G.edges])
    non_full_partitions = []

    for partition in partitioning:
        if len(partition) < 3:
            non_full_partitions.append(partition)
            if len(partition) == 2:
                ghz_edge = (partition[0],partition[1]) if partition[0] < partition[1] else (partition[1],partition[0])
                remaining_edges.discard(ghz_edge)
            continue
        for i in range(0, len(partition)):
            vertex_pos_map[partition[i]] = qubit_count
            qubit_count += 1
            if i > 0:
                ghz_edge = (partition[i-1],partition[i]) if partition[i-1] < partition[i] else (partition[i],partition[i-1])
                remaining_edges.discard(ghz_edge)
    # print("non full partitions",non_full_partitions)
    for non_full_partition in non_full_partitions:
        found_vertex = None
        found_neighbor = None
        for vertex in non_full_partition:
            non_partition_neighbor = set(G.neighbors(vertex)).difference(set(non_full_partition))
            if non_partition_neighbor:
                found_neighbor = non_partition_neighbor.pop()
                if found_neighbor in vertex_pos_map:
                    #there may exist the special case that there is no neighbor in a full partition, yet covering this case is more involved and we omit it for now
                    #actually this case happens, how to treat it?
                    found_vertex = vertex
                    break
        if not found_vertex:
            import pdb
            pdb.set_trace()
        # print("found neighbor",found_neighbor,"found vertex",found_vertex)    
        if len(non_full_partition) == 1:
            fusion_network.append((vertex_pos_map[found_neighbor], qubit_count))
            vertex_pos_map[found_neighbor] = qubit_count+1
            vertex_pos_map[found_vertex] = qubit_count+2
            qubit_count += 3
        else:
            fusion_network.append((vertex_pos_map[found_neighbor], qubit_count))
            fusion_network.append((qubit_count+2, qubit_count+3))
            vertex_pos_map[found_neighbor] = qubit_count+1
            other_vertex = [v for v in non_full_partition if v != found_vertex][0]
            vertex_pos_map[found_vertex] = qubit_count+4
            vertex_pos_map[other_vertex] = qubit_count+5
            # print("vertexposmap update found_neighbor",vertex_pos_map[found_neighbor],"found_vertex",vertex_pos_map[found_vertex])
            qubit_count += 6
        remaining_edges.discard((found_vertex, found_neighbor) if found_vertex < found_neighbor else (found_neighbor, found_vertex))

    # print("initial",vertex_pos_map, remaining_edges)
    for edge in remaining_edges:
        v1,v2 = (edge[0],edge[1]) if edge[0] < edge[1] else (edge[1],edge[0])
        # print("add edge",v1,v2)
        fusion_network.append((vertex_pos_map[v1], qubit_count))
        fusion_network.append((qubit_count+2, qubit_count+3))
        fusion_network.append((qubit_count+5,vertex_pos_map[v2]))
        vertex_pos_map[v1] = qubit_count+1
        vertex_pos_map[v2] = qubit_count+4
        # print("vertexposmap update v1",vertex_pos_map[v1],"v2",vertex_pos_map[v2])
        qubit_count += 6
    
    return fusion_network

def build_fusion_network_random(G: nx.Graph):
    """go with random order through the edges to build the fusion network. 
    We assign edge vertices to a new ghz if we have not encountered them before (non_full_partitions) 
    and fill the ghz up as soon as we encounter an adjacent edge. Otherwise we just have the normal procedures for cheap and expensive edges"""
    fusion_network = []
    vertex_pos_map = {}
    non_full_partitions = {}
    qubit_count = 0
    edgelist = list(G.edges)
    remaining_edges = [(v1,v2) if v1 < v2 else (v2,v1) for v1,v2 in edgelist]
    random.shuffle(remaining_edges)
    # print("rand edge",remaining_edges)
    #debug
    # remaining_edges = [(0, 1), (2, 3), (4, 5), (1, 4), (2, 4), (1, 6)]
    for edge in remaining_edges:
        v1,v2 = (edge[0],edge[1]) if edge[0] < edge[1] else (edge[1],edge[0])
        # print("edge:",v1,v2)
        #what to do if v1 and v2 are not present in vertex_pos_map? add ghz with one free qubit? 
        if not v1 in vertex_pos_map and not v2 in vertex_pos_map:
            # print("both not fixed")
            if v1 in non_full_partitions and v2 in non_full_partitions:
                # print("merge non full partitions")
                vertex_pos_map[non_full_partitions[v1]] = qubit_count
                vertex_pos_map[v1] = qubit_count+1
                vertex_pos_map[v2] = qubit_count+4
                vertex_pos_map[non_full_partitions[v2]] = qubit_count+5
                fusion_network.append((qubit_count+2,qubit_count+3))
                del non_full_partitions[non_full_partitions[v1]]
                del non_full_partitions[v1]
                del non_full_partitions[non_full_partitions[v2]]
                del non_full_partitions[v2]
                qubit_count += 6

            elif v1 in non_full_partitions:
                # print("add ",v2,"to non full partition",non_full_partitions[v1])
                vertex_pos_map[non_full_partitions[v1]] = qubit_count
                vertex_pos_map[v1] = qubit_count+1
                vertex_pos_map[v2] = qubit_count+2
                del non_full_partitions[non_full_partitions[v1]]
                del non_full_partitions[v1]
                qubit_count += 3
            elif v2 in non_full_partitions:
                # print("add ",v1,"to non full partition",non_full_partitions[v2])
                vertex_pos_map[non_full_partitions[v2]] = qubit_count
                vertex_pos_map[v2] = qubit_count+1
                vertex_pos_map[v1] = qubit_count+2
                del non_full_partitions[non_full_partitions[v2]]
                del non_full_partitions[v2]
                qubit_count += 3
            else:
                # print("init non full partition",v1,"->",v2)
                non_full_partitions[v1] = v2
                non_full_partitions[v2] = v1
        elif v1 in vertex_pos_map and not v2 in vertex_pos_map:
            # print("only",v1,"fixed")
            if v2 in non_full_partitions:
                vertex_pos_map[non_full_partitions[v2]] = qubit_count
                vertex_pos_map[v2] = qubit_count+1
                del non_full_partitions[non_full_partitions[v2]]
                del non_full_partitions[v2]
                fusion_network.append((qubit_count+2,qubit_count+3))
                fusion_network.append((qubit_count+5,vertex_pos_map[v1]))
                vertex_pos_map[v1] = qubit_count+4
                qubit_count += 6
            else:
                fusion_network.append((vertex_pos_map[v1],qubit_count))
                vertex_pos_map[v1] = qubit_count + 1
                vertex_pos_map[v2] = qubit_count + 2
                qubit_count += 3
        elif v2 in vertex_pos_map and not v1 in vertex_pos_map:
            # print("only",v2,"fixed")
            if v1 in non_full_partitions:
                vertex_pos_map[non_full_partitions[v1]] = qubit_count
                vertex_pos_map[v1] = qubit_count+1
                del non_full_partitions[non_full_partitions[v1]]
                del non_full_partitions[v1]
                fusion_network.append((qubit_count+2,qubit_count+3))
                fusion_network.append((qubit_count+5,vertex_pos_map[v2]))
                vertex_pos_map[v2] = qubit_count+4
                qubit_count += 6
            else:
                fusion_network.append((vertex_pos_map[v2],qubit_count))
                vertex_pos_map[v2] = qubit_count + 1
                vertex_pos_map[v1] = qubit_count + 2
                qubit_count += 3
        else:
            # print("both fixed")
            fusion_network.append((vertex_pos_map[v1], qubit_count))
            fusion_network.append((qubit_count+2,qubit_count+3))
            fusion_network.append((qubit_count+5,vertex_pos_map[v2]))
            vertex_pos_map[v1] = qubit_count + 1
            vertex_pos_map[v2] = qubit_count + 4
            qubit_count += 6
        # print("non_full_partitions:",non_full_partitions)
        # print("vertex_pos_map:",vertex_pos_map)
    return fusion_network


def draw_fusion_network(fusion_network):
    network = nx.Graph()
    max_qubit = int(max([q for f in fusion_network for q in f])/3)*3+3
    for i in range(0,max_qubit,3):
        network.add_edge(i,i+1,color='b')
        network.add_edge(i+1,i+2,color='b')
    for (q1,q2) in fusion_network:
        network.add_edge(q1,q2,color='r')
    nx.draw(network,edge_color= nx.get_edge_attributes(network,'color').values(),with_labels=True)
    return network

def draw_contracted_fusion_network(fusion_network):
    network = nx.Graph()
    max_qubit = int(max([q for f in fusion_network for q in f])/3)*3+3
    for i in range(0,max_qubit,3):
        network.add_edge(i,i+1)
        network.add_edge(i+1,i+2)
    for (q1,q2) in fusion_network:
        n1 = list(network.neighbors(q1))
        n2 = list(network.neighbors(q2))
        network.remove_nodes_from([q1,q2])
        for u in n1:
            for v in n2:
                network.add_edge(u,v)
    nx.draw(network,with_labels=True)
    return network