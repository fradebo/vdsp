from pyzx.graph.base import BaseGraph, VT
from pyzx.graph.graph_s import GraphS
from pyzx.simplify import to_gh, spider_simp
from pyzx.drawing import draw
from pyzx.utils import VertexType, EdgeType
from typing import Set, List

def p_condition(g: BaseGraph, x: VT, y: VT, p: VT, q: VT):
    # print("p cond for x:",x,"y:",y,"p:",p,"q:",q)
    # print("first ",(q in g.neighbors(p))," second ",(y in g.neighbors(p) and x in g.neighbors(q)))
    return (q in g.neighbors(p)) ^ (y in g.neighbors(p) and x in g.neighbors(q))

# Solves Problem 2: Given edges (x1, y1), (x2, y2) of G and a subset S of V(G) satisfying x1,y2 in S and x2,y1 notin S and |S| >= 2, 
# find, if there is one, a split {V1, V2} of G such that V1 is equal or a superset of S and x2,y1 are not in V1.
# A split is a partition {V1,V2} of the vertices in a graph, such that the edges between the sets form a complete bipartite graph 
# (each vertex from one border set is connected to every vertex in the other border set)
def find_split2(g: BaseGraph, s: Set[VT], x1: VT,y1: VT, x2: VT,y2: VT):
    t_curr = s.copy()
    s_curr = s.copy()
    while True:
        if len(t_curr) == 0:
            break
        p = t_curr.pop()
        for q in g.vertices():
            if (not q in s_curr) and (p_condition(g, x1, y1, p, q) or p_condition(g, x2, y2, q, p)):
                # print("update sets with",q)
                s_curr.add(q)
                t_curr.add(q)
    # print("final",s_curr)
    # print("final",t_curr)
    if len(s_curr) == g.num_vertices() - 1 or x2 in s_curr or y1 in s_curr:
        return None
    return (s_curr, g.vertex_set().difference(s_curr))

# Solves problem 1: Given edges (x1, y1), (x2, y2) of G, find, if there is one, a split {V1,V2} of G such that x1,y2 in V1 and x2,y1 in V2.
def find_split(g: BaseGraph, x1: VT,y1: VT, x2: VT,y2: VT):
    if x1 != y2:
        return find_split2(g, set([x1,y2]),x1,y1,x2,y2)
    # all the other cases may not be relevant, since we don't have graphs with multiedges
    elif x2 != y1:
        return find_split2(g, set([x2,y1]),x2,y2,x1,y1)
    else:
        assert(x1 == y2 and x2 == y1)
        z = g.vertex_set().difference(set([x1,y1])).pop()
        res = find_split2(g, set([x1,z]),x1,y1,x2,y2)
        if not res:
            res = find_split2(g, set([y1,z]),x2,y2,x1,y1)
        return res

def check_primeness_naive(g: BaseGraph):
    #run find_split with all possible edge combinations.
    for edge1 in g.edges():
        x1,y1 = g.edge_st(edge1)
        for edge2 in g.edges():
            x2,y2 = g.edge_st(edge2)
            if find_split(g, x1,y1,x2,y2):
                return False
            # all other cases due to undirectedness of our graphs
            elif find_split(g, x1,y1,y2,x2):
                return False
            elif find_split(g, y1,x1,x2,y2):
                return False
            elif find_split(g, y1,x1,y2,x2):
                return False
            else:
                continue
    return True

# More efficient check, since we are not running through all edges in m², but only through n² where n is the number of vertices
def check_primeness_spanning_tree(g: BaseGraph):
    root = g.vertex_set().pop()
    sp_tree = generate_spanning_tree(g, root)
    for edge1 in sp_tree.edges():
        x1,y1 = sp_tree.edge_st(edge1)
        for edge2 in sp_tree.edges():
            y2,x2 = sp_tree.edge_st(edge2)
            if find_split(g, x1,y1,x2,y2):
                return False
            # all other cases due to undirectedness of our graphs
            elif find_split(g, x1,y1,y2,x2):
                return False
            elif find_split(g, y1,x1,x2,y2):
                return False
            elif find_split(g, y1,x1,y2,x2):
                return False
            else:
                continue
    return True

# Even more efficient check for undirected graphs, we only need to interate once through the spanning tree
def check_primeness(g: BaseGraph):
    root = g.vertex_set().pop()
    sp_tree = generate_spanning_tree(g, root)
    for edge1 in sp_tree.edges():
        x1,y1 = sp_tree.edge_st(edge1)
        x2,y2 = (y1,x1)
        if find_split(g, x1,y1,x2,y2):
            return False
        # all other cases due to undirectedness of our graphs, not sure if we need them
        elif find_split(g, x1,y1,y2,x2):
            return False
        elif find_split(g, y1,x1,x2,y2):
            return False
        elif find_split(g, y1,x1,y2,x2):
            return False
        else:
            continue
    return True

# generates a spanning tree, i.e. a tree where all vertices are structured in a tree depending on the reachability starting from a root vertex
def generate_spanning_tree(g: BaseGraph, root: VT):
    tree = g.clone()
    all_edges = list(tree.edges())
    for edge in all_edges:
        # cleanup
        tree.remove_edge(edge)
    
    for neighbor in g.neighbors(root):
        tree.add_edge(tree.edge(root, neighbor))
    to_check = set(list(g.neighbors(root)))
    processed = set([root]).union(to_check)
    while to_check:
        next_check = set()
        for vertex in to_check:
            for neighbor in g.neighbors(vertex):
                if not neighbor in processed:
                    tree.add_edge(tree.edge(vertex, neighbor))
                    processed.add(neighbor)
                    next_check.add(neighbor)
        
        to_check = next_check
    return tree

# Given a split in two sets s1 and s2, returns the borders of the split, 
# i.e. the vertices which have an edge into the other set
def get_split_border(g: BaseGraph, s1: Set[VT], s2: Set[VT]):
    border1 = set()
    border2 = set()
    for edge in g.edges():
        x,y = g.edge_st(edge)
        if x in s1 and y in s2:
            border1.add(x)
            border2.add(y)
        elif y in s1 and x in s2:
            border1.add(y)
            border2.add(x)
    return (border1, border2)

# simple decomposition according to Bouchet 89, i.e. we introduce an edge with two endpoint vertices bridging the two split sets instead of a single vertex
def simple_decomposition(g: BaseGraph, s1: Set[VT], s2: Set[VT]):
    border1, border2 = get_split_border(g, s1, s2)
    #check for complete bipartite graph
    for vertex in border1:
        if not set(g.neighbors(vertex)).intersection(border2) == border2:
            print("split is no complete bipartite graph")
            assert(False)
    
    #edge removal
    qubit_average = 0 # for placing the new vertices correctly (useful for drawing the new graph)
    row_average = 0
    for vertex1 in border1:
        qubit_average += g.qubit(vertex1)
        row_average += g.row(vertex1)
        for vertex2 in border2:
            g.remove_edge(g.edge(vertex1,vertex2))
    qubit_average /= len(border1)
    row_average /= len(border1)
    # new vertices:
    v1 = g.add_vertex(qubit=qubit_average, row=row_average+0.33)
    v2 = g.add_vertex(qubit=qubit_average, row=row_average+0.66)
    
    #edge contraction
    g.add_edge(g.edge(v1,v2))
    for vertex in border1:
        g.add_edge(g.edge(vertex,v1), EdgeType.HADAMARD)
    for vertex in border2:
        g.add_edge(g.edge(vertex,v2), EdgeType.HADAMARD)
    
    return (v1,v2)

# simple decomposition according to Cunningham 82, i.e. single vertex as bridge
def simple_decomposition_cunningham(g: BaseGraph, s1: Set[VT], s2: Set[VT]):
    border1, border2 = get_split_border(g, s1, s2)
    #check for complete bipartite graph
    for vertex in border1:
        if not set(g.neighbors(vertex)).intersection(border2) == border2:
            print("split is no complete bipartite graph")
            assert(False)
    
    #edge removal
    qubit_average = 0 # for placing the new vertices correctly (useful for drawing the new graph)
    row_average = 0
    for vertex1 in border1:
        qubit_average += g.qubit(vertex1)
        row_average += g.row(vertex1)
        for vertex2 in border2:
            g.remove_edge(g.edge(vertex1,vertex2))
    qubit_average /= len(border1)
    row_average /= len(border1)
    # new vertices:
    v1 = g.add_vertex(VertexType.X, qubit=qubit_average, row=row_average+0.5)
    
    #edge contraction

    for vertex in border1:
        g.add_edge(g.edge(vertex,v1), EdgeType.HADAMARD)
    for vertex in border2:
        g.add_edge(g.edge(vertex,v1), EdgeType.HADAMARD)
    
    return v1

# generates spanning trees 
def generate_spanning_trees_from_split(g: BaseGraph, sp_tree: GraphS, s1: Set[VT], s2: Set[VT], v1: VT, v2: VT):

    sp_tree1 = sp_tree.clone()
    sp_tree1.remove_vertices(s2)
    sp_tree1.add_vertex_indexed(v1)
    for neighbor in g.neighbors(v1):
        if neighbor != v2:
            sp_tree1.add_edge(g.edge(neighbor, v1))

    sp_tree2 = sp_tree.clone()
    sp_tree2.remove_vertices(s1)
    sp_tree2.add_vertex_indexed(v2)
    for neighbor in g.neighbors(v2):
        if neighbor != v1:
            sp_tree2.add_edge(g.edge(neighbor, v2))
    
    return (sp_tree1, sp_tree2)

def generate_spanning_trees_from_split_cunningham(g: BaseGraph, sp_tree: GraphS, s1: Set[VT], s2: Set[VT], v: VT):
    r = list(s1)[0]
    g1 = g.clone()
    g1.remove_vertices(g.vertex_set().difference(s1.union(set([v]))))
    sp_tree1 = generate_spanning_tree(g1,r)

    sp_tree2 = sp_tree.clone()
    sp_tree2.remove_vertices(s1)
    sp_tree2.add_vertex_indexed(v)
    for neighbor in g.neighbors(v):
        if not neighbor in s1:
            sp_tree2.add_edge(g.edge(neighbor, v))
    
    return (sp_tree1, sp_tree2)


def is_star(g: BaseGraph):
    center = False
    for vertex in g.vertices():
        if len(g.neighbors(vertex)) > 1:
            if center:
                return False
            else:
                center = True
    return center

def is_prime(g: BaseGraph):
    return check_primeness(g)

def is_complete(g: BaseGraph):
    for vertex in g.vertices():
        if not g.vertex_set().difference(set(g.neighbors(vertex))) == set([vertex]):
            return False
    return True

def is_brittle(g: BaseGraph):
    return is_star(g) or is_complete(g)

def get_star_center(g: BaseGraph):
    center = None
    for vertex in g.vertices():
        if len(g.neighbors(vertex)) > 1:
            if center != None:
                return None
            else:
                center = vertex
    return center

def is_totally_decomposable(g: BaseGraph):
    g1 = g.clone()
    splits = prime_decomposition(g1)
    for split in splits:
        if len(split) != 3:
            return False
    return True

def check_proposition_5(g1: BaseGraph, g2: BaseGraph, v: Set[VT]):
    if g1.vertex_set().intersection(g2.vertex_set()) != v:
        return False
    if len(v) > 1:
        return False
    else:
        v = list(v)[0]
    if not (len(g1.vertices()) >= 3 and len(g2.vertices()) >= 3):
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

def prime_decomposition(g: BaseGraph):
    try:
        root = g.vertex_set().pop()
    except:
        return []
    sp_tree = generate_spanning_tree(g, root)
    return prime_decomposition_helper(g, sp_tree)

def prime_decomposition_helper(g: BaseGraph, sp_tree: BaseGraph):
    for edge1 in sp_tree.edges():
        x1,y1 = sp_tree.edge_st(edge1)
        x2,y2 = (y1,x1)
        g1 = g.clone()
        g1.remove_vertices(g.vertex_set().difference(sp_tree.vertex_set()))
        split = find_split(g1, x1,y1,x2,y2)
        if split:
            v = simple_decomposition_cunningham(g, split[0], split[1])
            # print("split: ",split[0], split[1], v)
            # draw(g, labels=True)
            sp_tree1, sp_tree2 = generate_spanning_trees_from_split_cunningham(g, sp_tree, split[0], split[1], v)


            splits1 = prime_decomposition_helper(g, sp_tree1)
            splits2 = prime_decomposition_helper(g, sp_tree2)
            return splits1+splits2
    return [sp_tree.vertex_set()]

def simple_composition_cunningham(g: BaseGraph, s1: Set[VT], s2: Set[VT], v: VT):
    border1 = set()
    border2 = set()
    for neighbor in g.neighbors(v):
        if not neighbor in s2:
            border1.add(neighbor)
        else:
            border2.add(neighbor)
        #TODO: can we be sure the split into borders is always correct?
    g.remove_vertex(v)
    for vertex1 in border1:
        for vertex2 in border2:
            g.add_edge(g.edge(vertex1, vertex2), EdgeType.HADAMARD)

def standard_decomposition(g: BaseGraph):
    splits = prime_decomposition(g)
    # print("splits: ",splits)
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
                    # print("found v",splits[idx], splits[idx2], list(v)[0])
                    g1 = g.clone()
                    g1.remove_vertices(g.vertex_set().difference(splits[idx]))
                    g2 = g.clone()
                    g2.remove_vertices(g.vertex_set().difference(splits[idx2]))
                    if check_proposition_5(g1,g2,v):
                        simple_composition_cunningham(g, splits[idx], splits[idx2], list(v)[0])
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


def get_subgraph(g: BaseGraph, subset: Set[VT]):
    res = g.clone()
    res.remove_vertices(g.vertex_set().difference(subset))
    return res

def is_correct(g: BaseGraph):
    v = get_star_center(g)
    if v:
        return g.type(v) == VertexType.Z
    return False


def get_all_marked_neighbors(g: BaseGraph, v: VT):
    marked_vertices = [vertex for vertex in g.neighbors(v) if g.type(v) == VertexType.X]
    res = marked_vertices.copy()
    while marked_vertices:
        marked_vertex = marked_vertices.pop()
        for neighbor in g.neighbors(marked_vertex):
            if not neighbor in res:
                marked_vertices.append(neighbor)
                res.append(neighbor)

    return res

# def get_adjacent_split_sets(g: BaseGraph, v: VT, splits):
#     marked_neighbors = get_all_marked_neighbors(g, v)
#     marked_sets = []
#     for split in splits:
#         for marked_neighbor in marked_neighbors:
#             if marked_neighbor in split and not v in split:
#                 marked_sets.append(split)
#                 break
    
#     return marked_sets

def get_adjacent_split_sets(g: BaseGraph, splits, lc_vertex: VT, closed):
    marked_vertices = [vertex for vertex in g.neighbors(lc_vertex) if g.type(vertex)==VertexType.X and not vertex in closed]
    res = []
    for marked_vertex in marked_vertices:
        for split in splits:
            if marked_vertex in split and not lc_vertex in split:
                res.append((marked_vertex,split))
                break

    return res

def local_complement(g: BaseGraph, lc_vertex, splits):
    print("lc on ",lc_vertex)
    complement_neighbors(g, list(g.neighbors(lc_vertex)))
    closed = [lc_vertex]
    candidates = set([lc_vertex])
    while candidates:
        candidate = candidates.pop()
        for marked_vertex, split_set in get_adjacent_split_sets(g, splits, candidate, closed):
            print("rec lc on ",marked_vertex)
            complement_neighbors(g, list(set(g.neighbors(marked_vertex)).intersection(split_set)))
            closed.append(marked_vertex)
            candidates.add(marked_vertex)
    
    draw(g, labels=True)

# def rec_lcomp_helper(g: BaseGraph, splits, marked_vertex, split_set):
#     complement_neighbors(g, split_set.difference(set([marked_vertex])))


def complement_neighbors(g: BaseGraph, vn: List[VT]):
    vn.sort()
    for n in vn:
        # flip edges
        for n2 in vn[vn.index(n)+1:]:
            if g.connected(n,n2):
                g.remove_edge(g.edge(n,n2))
            else:
                g.add_edge(g.edge(n,n2), EdgeType.HADAMARD)

# def local_complement(g: BaseGraph, v: VT, splits):
#     print("lc on ",v)
#     vn = list(g.neighbors(v))
#     complement_neighbors(g, vn)
    
#     # special lcomp for neighbors which are marked vertices
#     for split_set in get_adjacent_split_sets(g, v, splits):
#         complement_neighbors(g, split_set)
#     # marked_vertices = [vertex for vertex in vn if g.type(vertex) == VertexType.X]
#     # for marked_vertex in marked_vertices:
#     #     #recursive step
#     #     lcomp_marked_vertex(g, marked_vertex, set(vn).union(set([v])))

#     draw(g, labels=True)

# def lcomp_marked_vertex(g: BaseGraph, mv: VT, closed: Set[VT]):
#     print("lcomp recursion on",mv, closed)
#     complement_neighbors = list(set(g.neighbors(mv)).difference(closed))
#     # complement_neighbors = [vertex for vertex in split_set_neighbors if g.type(vertex) != VertexType.X]
#     complement_neighbors.sort()
#     for n in complement_neighbors:
#         # flip edges
#         for n2 in complement_neighbors[complement_neighbors.index(n)+1:]:
#             if g.connected(n,n2):
#                 g.remove_edge(g.edge(n,n2))
#             else:
#                 g.add_edge(g.edge(n,n2), EdgeType.HADAMARD)

#     marked_vertices = [vertex for vertex in complement_neighbors if g.type(vertex) == VertexType.X and not vertex in closed]
#     for marked_vertex in marked_vertices:
#         lcomp_marked_vertex(g, marked_vertex, set(complement_neighbors).union(set([mv])).union(closed))

# given the standard decomposition of a totally decomposable graph, 
# this brings the graph into a tree structure using a series of local complementation
def correct_components(g: BaseGraph, splits: List[Set[VT]]):
    complementations = []
    for component in splits:
        # print("component",component)
        subgraph = get_subgraph(g, component)
        if is_correct(subgraph):
            continue 
        if is_complete(subgraph):
            v = [vertex for vertex in component if g.type(vertex) == VertexType.Z][0]
            complementations.append(v)
            local_complement(g, v, splits)
            #This is no normal complement, we have to consider which vertices are adjacent in the original graph
        elif is_star(subgraph):
            m = get_star_center(subgraph)
            neighbor_component = [c for c in splits if c != component and m in c][0]
            nv = [vertex for vertex in neighbor_component if g.type(vertex) == VertexType.Z][0]
            complementations.append(nv)
            local_complement(g, nv, splits)
            v = [vertex for vertex in component if g.type(vertex) == VertexType.Z][0]
            complementations.append(v)
            local_complement(g, v, splits)
        else:
            # print("component",component,"is not correctable")
            continue
    return complementations
            # print("This should not happen, is the graph is lc equivalent to a tree and the decomposition canonical?")

def to_gl(g: BaseGraph):
    g1 = g.clone()
    to_gh(g1)
    spider_simp(g1, quiet=True)
    return g1
    #disentangle outputs
    

# converts a graph like diagram to a tree if possible
def graph_like_to_tree(g: BaseGraph):
    minputs = [list(g.neighbors(input))[0] for input in g.inputs()]
    moutputs = [list(g.neighbors(output))[0] for output in g.outputs()]
    g1 = g.clone()
    g1.remove_vertices(set(g.inputs()).union(set(g.outputs())))
    splits = standard_decomposition(g1)
    # TODO: This does not implement the quantum computing local complementation with phase updates in the spiders and changes of measurement planes!
    complementations = correct_components(g1, splits)
    recompose_graph(g1, splits)
    inputs = []
    for minput in minputs:
        input = g1.add_vertex()
        g1.add_edge(g1.edge(input, minput))
        inputs.append(input)
    g1.set_inputs(inputs)

    outputs = []
    for moutput in moutputs:
        output = g1.add_vertex()
        g1.add_edge(g1.edge(moutput, output))
        outputs.append(output)
    g1.set_outputs(outputs)

    return (g1,complementations)

def recompose_graph(g: BaseGraph, splits: List[Set]):
    if len(splits) < 2:
        return
    for idx in range(0,len(splits)):
        for idx2 in range(0, len(splits)):
            if idx == idx2:
                continue
            v = splits[idx].intersection(splits[idx2])
            if len(v) == 1:
                v = v.pop()
                if v in g.vertices():
                    border1 = set(list(g.neighbors(v))).intersection(splits[idx])
                    border2 = set(list(g.neighbors(v))).intersection(splits[idx2])
                    if len(border1) > 1 and len(border2) > 1:
                        import pdb
                        pdb.set_trace()     
                    g.remove_vertex(v)
                    for vertex1 in border1: 
                        for vertex2 in border2: 
                            g.add_edge(g.edge(vertex1, vertex2), EdgeType.HADAMARD)

