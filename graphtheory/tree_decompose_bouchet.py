import networkx as nx
from typing import List, Set

def p_condition(G: nx.Graph, x, y, p, q):
    # print("p cond for x:",x,"y:",y,"p:",p,"q:",q)
    # print("first ",(q in g.neighbors(p))," second ",(y in g.neighbors(p) and x in g.neighbors(q)))
    return (q in G.neighbors(p)) ^ (y in G.neighbors(p) and x in G.neighbors(q))


def find_split2(G: nx.Graph, S: Set, x1, y1, x2, y2):
    """Solves Problem 2: Given edges (x1, y1), (x2, y2) of G and a subset S of V(G) satisfying x1,y2 in S and x2,y1 notin S and |S| >= 2, 
    find, if there is one, a split {V1, V2} of G such that V1 is equal or a superset of S and x2,y1 are not in V1.
    A split is a partition {V1,V2} of the vertices in a graph, such that the edges between the sets form a complete bipartite graph 
    (each vertex from one border set is connected to every vertex in the other border set)"""
    t_curr = S.copy()
    s_curr = S.copy()
    while True:
        if len(t_curr) == 0:
            break
        p = t_curr.pop()
        for q in G.nodes:
            if (not q in s_curr) and (p_condition(G, x1, y1, p, q) or p_condition(G, x2, y2, q, p)):
                # print("update sets with",q)
                s_curr.add(q)
                t_curr.add(q)
    # print("final",s_curr)
    # print("final",t_curr)
    if len(s_curr) == len(G.nodes) - 1 or x2 in s_curr or y1 in s_curr:
        return None
    return (s_curr, set(G.nodes).difference(s_curr))


def find_split(G: nx.Graph, x1, y1, x2, y2):
    """Solves problem 1: Given edges (x1, y1), (x2, y2) of G, find, if there is one, a split {V1,V2} of G such that x1,y2 in V1 and x2,y1 in V2."""
    if x1 != y2:
        return find_split2(G, set([x1,y2]),x1,y1,x2,y2)
    # all the other cases may not be relevant, since we don't have graphs with multiedges
    elif x2 != y1:
        return find_split2(G, set([x2,y1]),x2,y2,x1,y1)
    else:
        assert(x1 == y2 and x2 == y1)
        z = set(G.nodes).difference(set([x1,y1])).pop()
        res = find_split2(G, set([x1,z]),x1,y1,x2,y2)
        if not res:
            res = find_split2(G, set([y1,z]),x2,y2,x1,y1)
        return res


def get_split_border(g: nx.Graph, s1: Set, s2: Set):
    """ Given a split in two sets s1 and s2, returns the borders of the split, 
        i.e. the vertices which have an edge into the other set"""
    border1 = set()
    border2 = set()
    for x,y in g.edges:
        if x in s1 and y in s2:
            border1.add(x)
            border2.add(y)
        elif y in s1 and x in s2:
            border1.add(y)
            border2.add(x)
    return (border1, border2)

def simple_decomposition_cunningham(G: nx.Graph, s1: Set, s2: Set):
    """simple decomposition according to Cunningham 82, i.e. single vertex as bridge"""
    border1, border2 = get_split_border(G, s1, s2)
    #check for complete bipartite graph
    for vertex in border1:
        if not set(G.neighbors(vertex)).intersection(border2) == border2:
            print("split is no complete bipartite graph")
            assert(False)
    
    #edge removal
    for vertex1 in border1:
        for vertex2 in border2:
            G.remove_edge(vertex1,vertex2)

    # new vertices:
    v1 = max(G.nodes)+1
    G.add_node(v1,marked=True) #careful, maybe labelling this does not work when we need to merge again.
    
    #edge contraction
    for vertex in border1:
        G.add_edge(vertex,v1)
    for vertex in border2:
        G.add_edge(vertex,v1)
    
    return v1

def simple_composition_cunningham(G: nx.Graph, s1: Set, s2: Set, v):
    border1 = set()
    border2 = set()
    for neighbor in list(G.neighbors(v)):
        if not neighbor in s2:
            border1.add(neighbor)
        else:
            border2.add(neighbor)
        #TODO: can we be sure the split into borders is always correct?
    G.remove_node(v)
    for vertex1 in border1:
        for vertex2 in border2:
            G.add_edge(vertex1, vertex2)

def generate_spanning_trees_from_split_cunningham(G: nx.Graph, sp_tree: nx.Graph, s1: Set, s2: Set, v):
    subgraph = nx.Graph(nx.induced_subgraph(G,s1.union(set([v]))))
    sp_tree1 = nx.random_spanning_tree(subgraph)

    sp_tree2 = sp_tree.copy()
    sp_tree2.remove_nodes_from(s1)
    sp_tree2.add_node(v)
    for neighbor in G.neighbors(v):
        if not neighbor in s1:
            sp_tree2.add_edge(neighbor, v)
    
    return (sp_tree1, sp_tree2)

def prime_decomposition_helper(G: nx.Graph, sp_tree: nx.Graph):
    for x1,y1 in sp_tree.edges:
        x2,y2 = (y1,x1)
        split = find_split(nx.Graph(nx.induced_subgraph(G,sp_tree.nodes)), x1,y1,x2,y2)
        if split:
            v = simple_decomposition_cunningham(G, split[0], split[1])
            # print("split: ",split[0], split[1], v)
            # draw(g, labels=True)
            sp_tree1, sp_tree2 = generate_spanning_trees_from_split_cunningham(G, sp_tree, split[0], split[1], v)


            splits1 = prime_decomposition_helper(G, sp_tree1)
            splits2 = prime_decomposition_helper(G, sp_tree2)
            return splits1+splits2
    return [set(sp_tree.nodes)]

def prime_decomposition(G: nx.Graph):
    sp_tree = nx.random_spanning_tree(G)
    return prime_decomposition_helper(G, sp_tree)

def check_proposition_5(g1: nx.Graph, g2: nx.Graph, v: int):
    if set(g1.nodes).intersection(set(g2.nodes)) != set([v]):
        return False

    if not (len(g1.nodes) >= 3 and len(g2.nodes) >= 3):
        return False 
    if is_complete(g1) and is_complete(g2):
        return True
    if is_star(g1) and is_star(g2):
        if get_star_center(g1) == v:
            return get_star_center(g2) != v
        else:
            return get_star_center(g2) == v
    #TODO: Do we neet the CTT check also for undirected graphs?
    return False

def standard_decomposition(G: nx.Graph):
    splits = prime_decomposition(G)
    if len(splits) <= 1:
        return splits

    while True:
        # find the standard decomposition by merging components together 
        # while maintaining the property of every component being brittle
        change = None
        for idx in range(0, len(splits)):
            for idx2 in range(idx+1, len(splits)):
                v = splits[idx].intersection(splits[idx2])
                if v:
                    # print("found v",splits[idx], splits[idx2], v)
                    v = v.pop()
                    if check_proposition_5(nx.Graph(nx.induced_subgraph(G,splits[idx])),nx.Graph(nx.induced_subgraph(G,splits[idx2])),v):
                        simple_composition_cunningham(G, splits[idx], splits[idx2], v)
                        change = (idx,idx2)
                        break
            if change:
                break
        if change:
            new_component = splits[change[0]].symmetric_difference(splits[change[1]])
            splits = [splits[i] for i in range(0,len(splits)) if not (i == idx or i == idx2)]
            splits.append(new_component)
        else:
            break
        
    return splits

def get_star_center(G: nx.Graph):
    center = None
    for node in G.nodes:
        if len(list(G.neighbors(node))) > 1:
            if center != None:
                return None
            else:
                center = node
    return center

def is_correct(G: nx.Graph):
    v = get_star_center(G)
    if v:
        return not 'marked' in G.nodes[v]
    return False

def is_complete(G: nx.Graph):
    for node in G.nodes:
        if not set(G.nodes).difference(set(G.neighbors(node))) == set([node]):
            return False
    return True

def is_star(G: nx.Graph):
    center = False
    for node in G.nodes:
        if len(list(G.neighbors(node))) > 1:
            if center:
                return False
            else:
                center = True
    return center

def get_adjacent_split_sets(G: nx.Graph, splits, lc_vertex, closed):
    marked_vertices = [node for node in G.neighbors(lc_vertex) if 'marked' in G.nodes[node] and not node in closed]
    res = []
    for marked_vertex in marked_vertices:
        for split in splits:
            if marked_vertex in split and not lc_vertex in split:
                res.append((marked_vertex,split))
                break

    return res

def complement_neighbors(G: nx.Graph, vn: List):
    vn.sort()
    for n in vn:
        # flip edges
        for n2 in vn[vn.index(n)+1:]:
            if n2 in G.neighbors(n):
                G.remove_edge(n,n2)
            else:
                G.add_edge(n,n2)

def local_complement(G: nx.Graph, lc_vertex, splits):
    # print("lc on ",lc_vertex)
    complement_neighbors(G, list(G.neighbors(lc_vertex)))
    closed = [lc_vertex]
    candidates = set([lc_vertex])
    while candidates:
        candidate = candidates.pop()
        for marked_vertex, split_set in get_adjacent_split_sets(G, splits, candidate, closed):
            # print("rec lc on ",marked_vertex)
            complement_neighbors(G, list(set(G.neighbors(marked_vertex)).intersection(split_set)))
            closed.append(marked_vertex)
            candidates.add(marked_vertex)

def correct_components(G: nx.Graph, splits: List[Set]):
    """given the standard decomposition of a totally decomposable graph, 
       this brings the graph into a tree structure using a series of local complementation"""
    complementations = []
    for component in splits:
        # print("component",component)
        subgraph = nx.Graph(nx.induced_subgraph(G,component))
        if is_correct(subgraph):
            continue 
        if is_complete(subgraph):
            v = [node for node in component if not 'marked' in G.nodes[node]][0]
            complementations.append(v)
            local_complement(G, v, splits)
            #This is no normal complement, we have to consider which vertices are adjacent in the original graph
        elif is_star(subgraph):
            m = get_star_center(subgraph)
            neighbor_component = [c for c in splits if c != component and m in c][0]
            nv = [node for node in neighbor_component if  not 'marked' in G.nodes[node]][0]
            complementations.append(nv)
            local_complement(G, nv, splits)
            v = [node for node in component if  not 'marked' in G.nodes[node]][0]
            complementations.append(v)
            local_complement(G, v, splits)
        else:
            # print("component",component,"is not correctable")
            continue
    return complementations

def recompose_graph(G: nx.Graph, splits: List[Set]):
    if len(splits) < 2:
        return
    for idx in range(0,len(splits)):
        for idx2 in range(0, len(splits)):
            if idx == idx2:
                continue
            v = splits[idx].intersection(splits[idx2])
            if len(v) == 1:
                v = v.pop()
                if v in G.nodes:
                    border1 = set(list(G.neighbors(v))).intersection(splits[idx])
                    border2 = set(list(G.neighbors(v))).intersection(splits[idx2])
                    if len(border1) > 1 and len(border2) > 1:
                        import pdb
                        pdb.set_trace()     
                    G.remove_node(v)
                    for vertex1 in border1: 
                        for vertex2 in border2: 
                            G.add_edge(vertex1, vertex2)

def is_totally_decomposable(G: nx.Graph):
    splits = prime_decomposition(G)
    for split in splits:
        if len(split) != 3:
            #not equivalent to a tree
            return False
    return True

def tree_decompose(G: nx.Graph):
    """main entry for bouchets tree decompose algorithm. If the graph is local equivalent (equivalent up to local complementation) to a tree, 
    the algorithm finds the series of local complementations and the equivalent tree in polynomial time"""
    if not is_totally_decomposable(G.copy()):
        return (G, [])
    splits = standard_decomposition(G)
    complementations = correct_components(G, splits)
    recompose_graph(G, splits)

    return (G,complementations)

if __name__ == "__main__":
    G = nx.Graph([(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(4,5),(6,7)])
    # nx.draw(G,with_labels=True)
    T, comp = tree_decompose(G)
    