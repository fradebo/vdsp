import perceval as pcvl
from compiler.PhotonicQubit import Qbit
from compiler.Graph import Tree, TreeNode
import perceval.components as symb
import numpy as np

class LOTree:

    def __init__(self, node):
        self.vertex_pos_map = {}
        self.circuit = pcvl.Circuit(6)
        self.tree = Tree(TreeNode(node))
    
    def add_edge(self, node1, node2):
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
            new_circuit = pcvl.Circuit(num_modes+12)
            if self.circuit._components:
                new_circuit.add(0, self.circuit, merge=True)
            fuse2(new_circuit,Qbit(num_modes+4),Qbit(num_modes+6))

            #update position map
            self.vertex_pos_map[node1] = num_modes+8
            self.vertex_pos_map[node2] = num_modes+10

        elif node1 in existing_nodes and not node2 in existing_nodes:
            new_circuit = pcvl.Circuit(num_modes+6)
            if self.circuit._components:
                new_circuit.add(0, self.circuit, merge=True)
            sink(new_circuit, Qbit(self.vertex_pos_map[node1]), Qbit(num_modes-2))
            fuse2(new_circuit,Qbit(num_modes-2),Qbit(num_modes))

            #update position map
            for k,v in self.vertex_pos_map.items():
                if v in range(self.vertex_pos_map[node1], num_modes-2):
                    self.vertex_pos_map[k] -= 2
            self.vertex_pos_map[node1] = num_modes+2
            self.vertex_pos_map[node2] = num_modes+4

        elif node2 in existing_nodes and not node1 in existing_nodes:
            new_circuit = pcvl.Circuit(num_modes+6)
            fuse2(new_circuit,Qbit(4),Qbit(6))
            if self.circuit._components:
                new_circuit.add(6, self.circuit, merge=True)

            #update position map
            for vertex in existing_nodes:
                self.vertex_pos_map[vertex] += 6
            self.vertex_pos_map[node1] = 8

        else:
            new_circuit = self.circuit
            sink(new_circuit, Qbit(self.vertex_pos_map[node1]), Qbit(self.vertex_pos_map[node2]-6))
            fuse2(new_circuit,Qbit(self.vertex_pos_map[node2]-6),Qbit(self.vertex_pos_map[node2]-4))

            #update position map
            for k,v in self.vertex_pos_map.items():
                if v in range(self.vertex_pos_map[node1], self.vertex_pos_map[node2]-6):
                    self.vertex_pos_map[k] -= 2
            self.vertex_pos_map[node1] = self.vertex_pos_map[node2]-2
        
        self.circuit = new_circuit
    
    def build_from_fusion_order(self, fusion_order):
        # input [(2,3), (5,6), ...] assume lower modes come first, 
        num_modes = int((max([v for fusion in fusion_order for v in fusion]) + 3) / 3) * 6 # num_modes always has to be divisible by 3
        self.circuit = pcvl.Circuit(num_modes)
        for mode in range(0,num_modes, 2):
            self.vertex_pos_map[int(mode/2)] = mode
        for m1,m2 in fusion_order:
            m1,m2 = (m1,m2) if m1 < m2 else (m2,m1)
            if m2-m1 == 1:
                fuse2(self.circuit, Qbit(self.vertex_pos_map[m1]), Qbit(self.vertex_pos_map[m2]))
            else:
                sink(self.circuit, Qbit(self.vertex_pos_map[m1]), Qbit(self.vertex_pos_map[m2]-2))
                fuse2(self.circuit, Qbit(self.vertex_pos_map[m2]-2), Qbit(self.vertex_pos_map[m2]))
                for k,v in self.vertex_pos_map.items():
                    if v in range(self.vertex_pos_map[m1], self.vertex_pos_map[m2]):
                        self.vertex_pos_map[k] -= 2
                self.vertex_pos_map[m1] = self.vertex_pos_map[m2]-2

    
    def merge(self, other, parent, child):
        """merges two LOTree objects. self is parent object, other the child object"""
        # merge circuits 
        # dbg_id = self.circuit.m-6+other.circuit.m
        # pcvl.pdisplay(self.circuit).save_png("drawingbug"+str(dbg_id)+'beforeself.png')
        # pcvl.pdisplay(other.circuit).save_png("drawingbug"+str(dbg_id)+'beforeother.png')
        new_circuit = pcvl.Circuit(self.circuit.m-6+other.circuit.m)
        if self.circuit._components:
            new_circuit.add(0,self.circuit, merge=True)
        if other.circuit._components:
            new_circuit.add(self.circuit.m-6,other.circuit, merge=True)
        self.circuit = new_circuit
        # pcvl.pdisplay(self.circuit).save_png("drawingbug"+str(dbg_id)+'after.png')
        # import pdb
        # pdb.set_trace()
        # merge trees
        if other.tree:
            self.tree.vertices += other.tree.vertices
            parent_idx = [i for i in range(len(self.tree.vertices)) if self.tree.vertices[i].value == parent][0]
            child_idx = [i for i in range(len(self.tree.vertices)) if self.tree.vertices[i].value == child][0]
            self.tree.vertices[child_idx].parent = self.tree.vertices[parent_idx]
            self.tree.vertices[parent_idx].children.append(self.tree.vertices[child_idx])


    
    def cdepth(self):
        """Given two photonic lines representing a qubit, we can calculate in which outer loop the last optical element on this two lines is positioned (cdepth = calculated depth).

        :return: it returns a list mapping every qubit index to the corresponding outer loop number defined above.
        :rtype: int
        """
        cdepth = {}
        depth, last = loopify(self.circuit,display=False)
        for v in range(0,self.circuit.m,2):
            ph = depth[v]
            pv = depth[v+1]

            assert(ph == pv)

            d = ph

            if d==0:
                d+=1

            cdepth[int(v/2)] = d

        return cdepth
    
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

def sink(circuit, q1:Qbit, q2:Qbit):
    
    """Sink the qubit q1 to the position of qubit q2

    :param q1:
    :type q1: Qbit
    :param q2:
    :type q2: Qbit
    """
    start = q1.pH.pos
    end = q2.pV.pos
    for i in range(start, end):
        circuit.add((i, i+1), symb.BS.H(np.pi))

    for i in range(start, end):
        circuit.add((i, i+1), symb.BS.H(np.pi))

def fuse2(circuit, q1:Qbit, q2:Qbit):
    """improved fuse2 as complete block"""
    m1 = q1.pH.pos
    m2 = q2.pH.pos
    assert(m1+2==m2)
    fusion_circ = pcvl.Circuit(4, name="Fuse2")
    fusion_circ.add((0, 1), symb.BS.H())
    fusion_circ.add((2, 3), symb.BS.H())
    fusion_circ.add((1, 2), symb.BS.H(np.pi))
    fusion_circ.add((0, 1), symb.BS.H())
    fusion_circ.add((2, 3), symb.BS.H())
    circuit.add(m1, fusion_circ, merge=False)

def fuse2_old(circuit, q1:Qbit, q2:Qbit):
    """Performs fusion of type 2 on qubits q1 and q2.

    :param q1:
    :type q1: Qbit
    :param q2:
    :type q2: Qbit
    """
    if q1.pH.pos > q2.pH.pos:
        pos1 = q2.pH.pos
        pos2 = q1.pH.pos
    else:
        pos1 = q1.pH.pos
        pos2 = q2.pH.pos
    
    # circuit.add((pos1, pos1+1), symb.BS.H(-np.pi))
    # circuit.add((pos2, pos2+1), symb.BS.H(-np.pi))
    circuit.add((pos1, pos1+1), symb.BS.H())
    circuit.add((pos2, pos2+1), symb.BS.H())

    for i in range(pos2-pos1-1):
        circuit.add((pos1+i+1, pos1+i+2), symb.BS.H(np.pi))

    for i in range(pos2-pos1-2):
        circuit.add((pos2-i-1, pos2-i), symb.BS.H(np.pi))

    # circuit.add((pos1, pos1+1), symb.BS.H(-np.pi))
    # circuit.add((pos2, pos2+1), symb.BS.H(-np.pi))
    circuit.add((pos1, pos1+1), symb.BS.H())
    circuit.add((pos2, pos2+1), symb.BS.H())


    q1.pH.set_witness(out_state=1)
    q1.pV.set_witness(out_state=0)
    
    q2.pH.set_witness(out_state=1)
    q2.pV.set_witness(out_state=0)

    return circuit._components[-2], circuit._components[-1] #return last hadamards for tracking fusion position


def convert_fusion_order(f_order):
    """converts a fusion order where ghz indexing is not in ascending order to an ascending order which we can then pass to the build_from_fusion_order method"""
    new_f_order = []
    vertex_map = {}
    current_max_vertex = 0
    for v1, v2 in f_order:
        for v in [v1,v2]:
            if not v in vertex_map:
                for i, v_ghz in enumerate(range(int(v/3)*3,(int(v/3)+1)*3)):
                    vertex_map[v_ghz] = current_max_vertex+i
                current_max_vertex += 3
        new_f_order.append((vertex_map[v1],vertex_map[v2]))
    return new_f_order


def loopify(circuit, display = False):
    """Function to rewrite the circuit in a format where the optical elements are divided accordingly to the corresponding outer loop they are performed in.

    :param display: if True, it adds graphical barriers to the Perceval circuit. It just makes the displaying more clear, defaults to True
    :type display: bool, optional
    :return: last[i] is the last optical element on the photonic line of index i, depth[i] is the outer loop number corresponding to last[i]
    :rtype: list, list
    """
    num_modes = circuit.m
    last = [None]*num_modes
    loop = {}
    for u in circuit._components:
        if (last[u[0][0]] == None) and (last[u[0][1]] == None):
            loop[u] = 0
        elif last[u[0][0]] == last[u[0][1]]:
            loop[u] = loop[last[u[0][0]]]
        else:
            l1 = 0
            l2 = 0
            if last[u[0][0]] != None:
                l1 = loop[last[u[0][0]]]
            if last[u[0][1]] != None:
                l2 = loop[last[u[0][1]]]
        
            l = max(l1, l2+1)
        
            loop[u] = l

        last[u[0][0]] = last[u[0][1]] = u


    depth = [0]*num_modes
    for p in range(num_modes):
        if last[p] != None:
            depth[p] = loop[last[p]]
        else:
            depth[p] = 0

    return depth, last


def build_optimal_linear(node, lotree, reversed=False):
    """Build the optimal DFS-ordered circuit on the object qtree. The idea is to order each vertex's children, recursively computing the weight of the subtrees. To compute the weight of the subtree we lunch the function on a newly created QTree object.

    :param node: the head of the subtree we are building
    :type node: TreeNode
    :param qtree: the QTree object in which we are building the circuit
    :type qtree: QTree
    :return: the number of outer loops needed to build the subtree with head the TreeNode head. We compute it 
    :rtype: int
    if reversed we get the worst dfs order instead
    """

    children = [(x, build_optimal_linear(x, LOTree(x.value), reversed)) for x in node.children]
    children.sort(key=lambda x: x[1][1], reverse=reversed)

    for child, childtree in children:
        lotree.add_edge(node.value, child.value)
        lotree.merge(childtree[0], node.value, child.value)

    return lotree, max(lotree.cdepth().values())
