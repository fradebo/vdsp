import perceval as pcvl
from compiler.PhotonicQubit import Qbit
from compiler.Graph import Tree, TreeNode
import perceval.components as symb
import numpy as np

class LOTree:

    def __init__(self, node):
        self.vertex_pos_map = {}
        self.circuit = pcvl.Circuit(4)
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
        print("existing nodes",existing_nodes)

        if not (node1 in existing_nodes or node2 in existing_nodes):
            print("case1")
            new_circuit = pcvl.Circuit(num_modes+8)
            new_circuit.add(0, self.circuit, merge=True)
            fuse(new_circuit,Qbit(num_modes+2),Qbit(num_modes+4))

            #update position map
            self.vertex_pos_map[node1] = num_modes+4
            self.vertex_pos_map[node2] = num_modes+6

        elif node1 in existing_nodes and not node2 in existing_nodes:
            print("case2")
            new_circuit = pcvl.Circuit(num_modes+4)
            new_circuit.add(0, self.circuit, merge=True)
            sink(new_circuit, Qbit(self.vertex_pos_map[node1]), Qbit(num_modes-2))
            fuse(new_circuit,Qbit(num_modes-2),Qbit(num_modes))

            #update position map
            for k,v in self.vertex_pos_map.items():
                if v in range(self.vertex_pos_map[node1], num_modes-2):
                    self.vertex_pos_map[k] -= 2
            self.vertex_pos_map[node1] = num_modes
            self.vertex_pos_map[node2] = num_modes+2

        elif node2 in existing_nodes and not node1 in existing_nodes:
            print("case3")
            new_circuit = pcvl.Circuit(num_modes+4)
            fuse(new_circuit,Qbit(2),Qbit(4))
            new_circuit.add(4, self.circuit, merge=True)

            #update position map
            for vertex in existing_nodes:
                self.vertex_pos_map[vertex] += 4
            self.vertex_pos_map[node1] = 4

        else:
            print("case4")
            new_circuit = self.circuit
            sink(new_circuit, Qbit(self.vertex_pos_map[node1]), Qbit(self.vertex_pos_map[node2]-4))
            fuse(new_circuit,Qbit(self.vertex_pos_map[node2]-4),Qbit(self.vertex_pos_map[node2]-2))

            #update position map
            for k,v in self.vertex_pos_map.items():
                if v in range(self.vertex_pos_map[node1], self.vertex_pos_map[node2]-4):
                    self.vertex_pos_map[k] -= 2
            self.vertex_pos_map[node1] = self.vertex_pos_map[node2]-2
        
        self.circuit = new_circuit
    
    
    def merge(self, other, parent, child):
        """merges two LOTree objects. self is parent object, other the child object"""
        # merge circuits 
        new_circuit = pcvl.Circuit(self.circuit.m-4+other.circuit.m)
        new_circuit.add(0,self.circuit, merge=True)
        new_circuit.add(self.circuit.m-4,other.circuit, merge=True)
        self.circuit = new_circuit
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
    
    def fusion_order(self):
        order = []
        expand = [(self.tree.head,0)]
        while expand:
            vertex,depth = expand.pop()
            order.append(vertex.value)
            for child in reversed(vertex.children):
                expand.append((child, depth+1))
        return order

def sink(circuit, q1:Qbit, q2:Qbit):
    print("sink",q1.pH.pos,q2.pH.pos)
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
    
    circuit.add((pos1, pos1+1), symb.BS.H(-np.pi))
    circuit.add((pos2, pos2+1), symb.BS.H(-np.pi))

    for i in range(pos2-pos1-1):
        circuit.add((pos1+i+1, pos1+i+2), symb.BS.H(np.pi))

    for i in range(pos2-pos1-2):
        circuit.add((pos2-i-1, pos2-i), symb.BS.H(np.pi))

    circuit.add((pos1, pos1+1), symb.BS.H(-np.pi))
    circuit.add((pos2, pos2+1), symb.BS.H(-np.pi))


    q1.pH.set_witness(out_state=1)
    q1.pV.set_witness(out_state=0)
    
    q2.pH.set_witness(out_state=1)
    q2.pV.set_witness(out_state=0)

def fuse(circuit, q1:Qbit, q2:Qbit):
    """Perform fusion of type 1 on the qubit q1 and q2, witnessing the qubit q1.

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

    for i in range(pos2-pos1):
        circuit.add((pos1+i, pos1+i+1), symb.BS.H(np.pi))

    for i in range(pos2-pos1-1):
        circuit.add((pos2-i-2, pos2-i-1), symb.BS.H(np.pi))

    circuit.add((q1.pH.pos, q1.pV.pos), symb.BS.H())

    q1.pH.set_witness(out_state=1)
    q1.pV.set_witness(out_state=0)


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
