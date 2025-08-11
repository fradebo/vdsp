import networkx as nx

def ghz_partitioning_heuristic(G: nx.Graph):
    """Heuristic to find a graph partitioning such that every partition is connected and has cardinality of at most 3. 
    The desired result is to be minimal wrt. to the number of partitions, that is almost every partition should have cardinality 3.
    Getting the optimum is an NP-hard problem, see for instance arXiv:1910.02470v1, our heuristic tries the following:
    At each iteration we identify vertices which have only a single neighbor not belonging to any partition; 
    if the vertex has other neighbors which are in a partition of cardinality <=2, we add it to this partition, else 
    we add a new partition of either only the vertex itself, or the vertex and a non processed neighbor. 
    If there are no unary vertices, we either extend existing partitionings with their lowest degree neighbor
    or we add a new partition with the highest degree vertex of the unprocessed vertices
    TODO: make sure that middle vertex of any returned ghz is indeed connected to the other vertices"""
    vertex_degree_map = {v: G.degree[v] for v in G.nodes}
    initial_ghzs = dict()
    ghz_count = 0
    vertex_to_ghz_map = dict()
    while True:
        for k, v in vertex_degree_map.items():
            vertex_degree_map[k] = len([n for n in G.neighbors(k) if not n in vertex_to_ghz_map.keys()])
        # print("vertex deg map",vertex_degree_map)
        #update vertex_degree_map
        if not vertex_degree_map.keys():
            break
        #adjust node degrees based on which vertices are still not mapped
        unaries = [v for v in vertex_degree_map.keys() if vertex_degree_map[v] == 1]
        if unaries:
            #check unary vertices
            v = unaries[0]
            processed = False
            for neighbor in list(G.neighbors(v)):
                if neighbor in vertex_degree_map.keys():
                    ghz_count += 1
                    initial_ghzs[ghz_count] = [v, neighbor]
                    vertex_to_ghz_map[v] = ghz_count
                    vertex_to_ghz_map[neighbor] = ghz_count
                    del vertex_degree_map[v]
                    del vertex_degree_map[neighbor]
                    processed = True
                    break
                else:
                    ghz_id = vertex_to_ghz_map[neighbor]
                    if len(initial_ghzs[ghz_id]) < 3:
                        initial_ghzs[ghz_id].append(v)
                        vertex_to_ghz_map[v] = ghz_id
                        del vertex_degree_map[v]
                        processed = True
                        break
            if not processed:
                #unfortunately a singular vertex then
                ghz_count += 1
                initial_ghzs[ghz_count] = [v]
                vertex_to_ghz_map[v] = ghz_count
                del vertex_degree_map[v]
        
        else:
            # if there are no unary candidates left:
            # first try to complete initial ghzs with already two vertices
            two_ary_ghzs = [ghz for ghz in initial_ghzs.values() if len(ghz) == 2]
            if two_ary_ghzs:
                v1,v2 = two_ary_ghzs[0]
                ghz_id = vertex_to_ghz_map[v1]
                #get all neighbors
                neighbors = set(G.neighbors(v1)).union(set(G.neighbors(v2))).intersection(set(vertex_degree_map.keys()))
                if neighbors:
                    lowest_neighbor = sorted(list(neighbors),key= lambda k: vertex_degree_map[k])[0]
                    initial_ghzs[ghz_id].append(lowest_neighbor)
                    vertex_to_ghz_map[lowest_neighbor] = ghz_id
                    del vertex_degree_map[lowest_neighbor]
                    continue
            # then try to append to an initial ghz with only one vertex
            unary_ghzs = [ghz for ghz in initial_ghzs.values() if len(ghz) == 1]
            if unary_ghzs:
                v1 = unary_ghzs[0][0]
                ghz_id = vertex_to_ghz_map[v1]
                #get all neighbors
                neighbors = set(G.neighbors(v1)).intersection(set(vertex_degree_map.keys()))
                if neighbors:
                    lowest_neighbor = sorted(list(neighbors),key= lambda k: vertex_degree_map[k])[0]
                    initial_ghzs[ghz_id].append(lowest_neighbor)
                    vertex_to_ghz_map[lowest_neighbor] = ghz_id
                    del vertex_degree_map[lowest_neighbor]
                    continue
            
            #default: add new ghz with highest degree vertex
            highest_degree_vertex = max(vertex_degree_map, key=vertex_degree_map.get)
            ghz_count += 1
            initial_ghzs[ghz_count] = [highest_degree_vertex]
            vertex_to_ghz_map[highest_degree_vertex] = ghz_count
            del vertex_degree_map[highest_degree_vertex]

    #optional: sort ghzs inside so highest degree vertex is in the middle
    for k, ghz in initial_ghzs.items():
        if len(ghz) == 3:
            middle_vert = sorted([v for v in ghz if len(set(G.neighbors(v)).intersection(set(ghz))) == 2],key=lambda v: G.degree[v])[-1]
            middle_vert_idx = initial_ghzs[k].index(middle_vert)
            if middle_vert_idx != 1:
                initial_ghzs[k][middle_vert_idx], initial_ghzs[k][1] = initial_ghzs[k][1], middle_vert #swap
    return list(initial_ghzs.values())