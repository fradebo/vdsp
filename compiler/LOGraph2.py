import perceval as pcvl
from compiler.Graph import Tree, TreeNode
import perceval.components as symb
import numpy as np
import networkx as nx

class LOGraph:
    """Class for building graph states with a perceval circuit
    currently seems to work with build_optimal_dfs, but graph results suboptimal since sometimes graph edges interfere with good orderings
    i.e. some modes are redundantly lifted just to be sinked afterwards."""
    def __init__(self, tree_head, graph: nx.Graph):
        self.vertex_pos_map = {}
        self.circuit = pcvl.Circuit(6)
        self.tree = Tree(TreeNode(tree_head))
        self.graph = graph
    
    def add_tree_edge(self, node1, node2):
        print("add tree edge",node1,node2)
        """Adds a tree edge to the linear optics circuit distinguishing between four cases: 
        - if none of the nodes are present, we add two ghzs at the end and fuse them
        - if only the parent node is present, we add the child at the end, sink the parent and fuse
        - if only the child is present (assumed to be at the first six modes), we insert the parent at the beginning and fuse
        - if both are present, do not increase the circuit but just sink the parent and fuse"""

        if not self.vertex_pos_map:
            num_modes = 0
        else:
            num_modes = self.circuit.m
        
        existing_nodes = self.vertex_pos_map.keys()

        if not (node1 in existing_nodes or node2 in existing_nodes):
            self.enlarge_circuit(6)
            self.fuse2(num_modes+4,num_modes+6)

            #update position map
            self.vertex_pos_map[node1] = num_modes+8
            self.vertex_pos_map[node2] = num_modes+10

        elif node1 in existing_nodes and not node2 in existing_nodes:
            self.enlarge_circuit(6)
            self.sink(self.vertex_pos_map[node1],num_modes-2)
            self.fuse2(num_modes-2,num_modes)

            self.vertex_pos_map[node1] = num_modes+2
            self.vertex_pos_map[node2] = num_modes+4

        elif node2 in existing_nodes and not node1 in existing_nodes:
            self.enlarge_circuit(6,True)
            self.fuse2(4,6)

            #update position map
            for vertex in existing_nodes:
                self.vertex_pos_map[vertex] += 6
            self.vertex_pos_map[node1] = 8

        else:
            #both nodes are present; can happen if we don't build up in top down approach
            node1, node2 = (node1,node2) if self.vertex_pos_map[node1] < self.vertex_pos_map[node2] else (node2,node1) # ensure correct order
            self.sink(self.vertex_pos_map[node1], self.vertex_pos_map[node2]-6)
            self.fuse2(self.vertex_pos_map[node2]-6,self.vertex_pos_map[node2]-4)

            self.vertex_pos_map[node1] = self.vertex_pos_map[node2]-2 # could be a problem here, do we really involve node2 in the fusion?
        
        print(self.vertex_pos_map)
        # pcvl.pdisplay(self.circuit).save_png('tree_edge'+str(node1)+str(node2)+'.png')

    
    def add_graph_edge(self, node1, node2):
        print("add graph edge",node1,node2)
        num_modes = self.circuit.m
        self.enlarge_circuit(12)
        self.fuse2(num_modes+4,num_modes+6)
        node1, node2 = (node1,node2) if self.vertex_pos_map[node1] < self.vertex_pos_map[node2] else (node2,node1) # ensure correct order
        self.sink(self.vertex_pos_map[node2], num_modes+8)
        self.sink(self.vertex_pos_map[node1], num_modes-4)
        self.fuse2(num_modes-4, num_modes-2)
        self.fuse2(num_modes+8, num_modes+10)
        self.vertex_pos_map[node1] = num_modes
        self.vertex_pos_map[node2] = num_modes+6
        print(self.vertex_pos_map)
        # pcvl.pdisplay(self.circuit).save_png('graph_edge'+str(node1)+str(node2)+'.png')

    
    def sink(self, start_mode, end_mode):
        print("sink from ",start_mode,"to",end_mode)
        if start_mode == end_mode:
            return
        for i in range(start_mode, end_mode+1):
            self.circuit.add((i, i+1), symb.BS.H(np.pi))

        for i in range(start_mode, end_mode+1):
            self.circuit.add((i, i+1), symb.BS.H(np.pi))
        
        #update position map
        for k,v in self.vertex_pos_map.items():
            if v in range(start_mode, end_mode+2):
                self.vertex_pos_map[k] -= 2
    
    def fuse2(self, m1, m2):
        print("fuse",m1,m2)
        assert(m1+2==m2)
        fusion_circ = pcvl.Circuit(4, name="Fuse2")
        fusion_circ.add((0, 1), symb.BS.H())
        fusion_circ.add((2, 3), symb.BS.H())
        fusion_circ.add((1, 2), symb.BS.H(np.pi))
        fusion_circ.add((0, 1), symb.BS.H())
        fusion_circ.add((2, 3), symb.BS.H())
        self.circuit = self.circuit.add(m1, fusion_circ, merge=False)
    
    def enlarge_circuit(self,num_modes, insert_above=False):

        new_circuit = pcvl.Circuit(self.circuit.m + num_modes)
        if self.circuit._components:
            if insert_above:
                new_circuit.add(num_modes, self.circuit, merge=True)
            else:
                new_circuit.add(0, self.circuit, merge=True)
        self.circuit = new_circuit

    def merge(self, other, parent, child):
        print("merge",parent,child)
        orig_num_modes = self.circuit.m
        # why the -6?
        new_circuit = pcvl.Circuit(orig_num_modes-6+other.circuit.m)
        if self.circuit._components:
            new_circuit.add(0,self.circuit, merge=True)
        if other.circuit._components:
            new_circuit.add(orig_num_modes-6,other.circuit, merge=True)
        self.circuit = new_circuit

        for k,v in other.vertex_pos_map.items():
            self.vertex_pos_map[k] = v + orig_num_modes-6

        if other.tree:
            self.tree.vertices += other.tree.vertices
            parent_idx = [i for i in range(len(self.tree.vertices)) if self.tree.vertices[i].value == parent][0]
            child_idx = [i for i in range(len(self.tree.vertices)) if self.tree.vertices[i].value == child][0]
            self.tree.vertices[child_idx].parent = self.tree.vertices[parent_idx]
            self.tree.vertices[parent_idx].children.append(self.tree.vertices[child_idx])
        
        print(self.vertex_pos_map)
        # pcvl.pdisplay(self.circuit).save_png('merge'+str(parent)+str(child)+'.png')


    def get_mode_count(self):
        """we track the outer loop number of modes, i.e. for determining the fusion order"""
        mode_count = [0]*self.circuit.m
        for component in self.circuit._components:
            if len(component[0]) > 2:
                # skip annoying identity gate at beginning and fusion gates (outsourced out of the loop)
                continue
            mode0, mode1 = component[0]
            outer_loop = max(mode_count[mode0],mode_count[mode1])
            mode_count[mode0] = outer_loop + 1
            mode_count[mode1] = outer_loop

        return mode_count
    
    def depth(self):
        return max(self.get_mode_count())
    
    def get_fusion_order(self):
        mode_count = [0]*self.circuit.m
        pos_map = {i: i for i in range(self.circuit.m)} #end_mode -> start_mode
        fusion_component_depth = {} #start_modes -> depth
        for component in self.circuit._components:
            if len(component[0]) > 2 and component[1].name != 'Fuse2':
                # skip annoying identity gate at beginning
                continue
            elif component[1].name == 'Fuse2':
                start_modes = tuple([pos_map[m] for m in component[0]])
                fusion_component_depth[start_modes] = max([mode_count[m] for m in component[0]])
            else:
                mode0, mode1 = component[0]
                outer_loop = max(mode_count[mode0],mode_count[mode1])
                mode_count[mode0] = outer_loop + 1
                mode_count[mode1] = outer_loop
                
                temp = pos_map[mode0]
                pos_map[mode0] = pos_map[mode1]
                pos_map[mode1] = temp

        reduced_fusion_list = [(v,(k[0],k[2])) for k,v in fusion_component_depth.items()]
        return [(int(m[0]/2),int(m[1]/2)) for l,m in sorted(reduced_fusion_list)]
    
    def build_from_fusion_order(self, fusion_order):
        # input [(2,3), (5,6), ...] assume lower modes come first, 
        num_modes = int((max([v for fusion in fusion_order for v in fusion]) + 3) / 3) * 6 # num_modes always has to be divisible by 3
        self.circuit = pcvl.Circuit(num_modes)
        for mode in range(0,num_modes, 2):
            self.vertex_pos_map[int(mode/2)] = mode
        for m1,m2 in fusion_order:
            m1,m2 = (m1,m2) if m1 < m2 else (m2,m1)
            if m2-m1 == 1:
                self.fuse2(self.vertex_pos_map[m1], self.vertex_pos_map[m2])
            else:
                self.sink(self.vertex_pos_map[m1],self.vertex_pos_map[m2]-2)
                self.fuse2(self.vertex_pos_map[m2]-2, self.vertex_pos_map[m2])

                self.vertex_pos_map[m1] = self.vertex_pos_map[m2]-2


def build_optimal_dfs_search(node, lograph: LOGraph, reverse=False):
    """Build the optimal DFS-ordered circuit on the object qtree. The idea is to order each vertex's children, recursively computing the weight of the subtrees. To compute the weight of the subtree we lunch the function on a newly created QTree object.

    :param node: the head of the subtree we are building
    :type node: TreeNode
    :param qtree: the QTree object in which we are building the circuit
    :type qtree: QTree
    :return: the number of outer loops needed to build the subtree with head the TreeNode head. We compute it 
    :rtype: int
    if reversed we get the worst dfs order instead
    """

    children = [(x, build_optimal_dfs_search(x, LOGraph(x.value, lograph.graph), reverse)) for x in node.children]
    children.sort(key=lambda x: x[1][1], reverse=reverse)
    # first add all tree edges in correct ordering
    for child, childtree in children:
        lograph.add_tree_edge(node.value, child.value)
        lograph.merge(childtree[0], node.value, child.value)
    
    # then check for missing graph edges and add them (no 'correct' ordering possible here)
    graph_vertices = [node.value]
    for child, childtree in children:
        subtree_vertices = [v.value for v in childtree[0].tree.vertices]
        print("subtree with head",child.value,"vertices:",subtree_vertices)
        #add graph edges if necessary.
        for edge in lograph.graph.edges:
            if edge[0] in graph_vertices and edge[1] in subtree_vertices and not (edge[1] == child.value and edge[0] == node.value):
                lograph.add_graph_edge(edge[0],edge[1])
            elif edge[1] in graph_vertices and edge[0] in subtree_vertices and not (edge[0] == child.value and edge[1] == node.value):
                lograph.add_graph_edge(edge[0],edge[1])
        graph_vertices += subtree_vertices

    return lograph, lograph.depth()

def random_order(tree_head, lograph):
    """we want to get a fusion order to build a graph without build optimal"""
    