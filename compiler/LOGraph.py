"""This file should contain all necessary tools to build a circuit of GHZ as basis states given a fusion order of the form List[Tuple[int,int]]"""
import perceval as pcvl
import perceval.components as symb
from typing import Tuple, Dict, List
from compiler.LOTree import fuse2, loopify
from compiler.Graph import Tree, TreeNode
from compiler.PhotonicQubit import Qbit
from scripts.random_graph_order import get_spanning_tree
import numpy as np
import networkx as nx

class LOGraph:
    def __init__(self, ghz_mapping: List[List[int]], head: int):
        pos = 0
        self.vertex_pos_map = {}
        self.fusion_pos_map = {} #edge to component
        self.ghz_mapping = ghz_mapping
        self.edges = []
        for ghz in ghz_mapping:
            for i,vertex in enumerate(ghz):
                if head in ghz:
                    self.vertex_pos_map[vertex] = pos
                    pos += 2
                if i > 0:
                    self.edges.append(sorted([ghz[i-1],vertex]))
        self.circuit = pcvl.Circuit(6)
    
    def add_ghz_of_vertex(self, vertex: int):
        for ghz in self.ghz_mapping:
            if vertex in ghz:
                new_circuit = pcvl.Circuit(self.circuit.m+6)
                new_circuit.add(0, self.circuit, merge=True)
                for i,v in enumerate(ghz):
                    self.vertex_pos_map[v] = self.circuit.m + 2*i
                self.circuit = new_circuit
    
    def sink(self, start_mode, end_mode):
        if start_mode == end_mode:
            return
        for i in range(start_mode, end_mode+1):
            self.circuit.add((i, i+1), symb.BS.H(np.pi))

        for i in range(start_mode, end_mode+1):
            self.circuit.add((i, i+1), symb.BS.H(np.pi))
        
        for k,v in self.vertex_pos_map.items():
            if v in range(start_mode, end_mode):
                self.vertex_pos_map[k] -= 2

    def add_edge(self, edge: Tuple[int,int]):
        # print("add edge",edge)
        edge = sorted(list(edge))
        if edge in self.edges:
            return
        v1, v2 = edge

        if not v1 in self.vertex_pos_map.keys():
            self.add_ghz_of_vertex(v1)
        
        if not v2 in self.vertex_pos_map.keys():
            self.add_ghz_of_vertex(v2)
        
        old_size = self.circuit.m
        new_circuit = pcvl.Circuit(old_size+8)
        new_circuit.add(0, self.circuit, merge=True)
        self.circuit = new_circuit

        if self.vertex_pos_map[v1] > self.vertex_pos_map[v2]:
            self.sink(self.vertex_pos_map[v1], old_size+4)
            self.sink(self.vertex_pos_map[v2], old_size-4)

            self.vertex_pos_map[v1] = old_size+2
            self.vertex_pos_map[v2] = old_size
        else:
            self.sink(self.vertex_pos_map[v2],old_size+4)
            self.sink(self.vertex_pos_map[v1],old_size-4)
            
            self.vertex_pos_map[v2] = old_size+2
            self.vertex_pos_map[v1] = old_size

        c1, c2 = fuse2(self.circuit,Qbit(old_size-4),Qbit(old_size-2)) #TODO: rewrite this so no actual fusion happens but we still can trace the outer loop in which we would fuse.
        c3, c4 = fuse2(self.circuit,Qbit(old_size+4),Qbit(old_size+6))
        self.fusion_pos_map[tuple(sorted([v1,v2]))] = (c1,c2,c3,c4)

        

    def get_fusion_order(self):
        """how can we determine a fusion order if the two fusions for a complicated edge are not strictly after each other,
        i.e. or interleaved with other operations? Don't care for the moment?"""
        hadamards = self.get_hadamard_positions()
        edge_order_dict = {}
        for edge, components in self.fusion_pos_map.items():
            edge_order_dict[edge] = (max([hadamards[c] for c in components]), max([c[0][1] for c in components]))
        print(edge_order_dict)
        #sort needs to account for both outer_loop and mode number of the last fusion mode.
        return sorted(edge_order_dict.keys(), key=lambda edge: edge_order_dict[edge])

    def depth(self):
        """returns the number of outer loops required to implement a given circuit in the double loop architecture
        similar to loopify, but loopify seems to be wrong in some cases + easier calculation of circuit depth in a loop"""
        mode_count = [0]*self.circuit.m
        for component in self.circuit._components:
            if len(component[0]) > 2:
                # skip annoying identity gate at beginning
                continue
            mode0, mode1 = component[0]
            outer_loop = max(mode_count[mode0],mode_count[mode1])
            mode_count[mode0] = outer_loop + 1
            mode_count[mode1] = outer_loop
            # print("component",component,"can be processed in outer loop no",outer_loop)

        return max(mode_count)
    
    def get_hadamard_positions(self):
        """similar to loopify3, but we track the outer loop number of all hadamards , i.e. for determining the fusion order"""
        hadamards = {}
        mode_count = [0]*self.circuit.m
        for component in self.circuit._components:
            if len(component[0]) > 2:
                # skip annoying identity gate at beginning
                continue
            mode0, mode1 = component[0]
            outer_loop = max(mode_count[mode0],mode_count[mode1])
            mode_count[mode0] = outer_loop + 1
            mode_count[mode1] = outer_loop
            if 'theta' in component[1]._params and component[1]._params['theta']._value < 3:
                hadamards[component] = outer_loop

        return hadamards


class LOGraph2:
    def __init__(self, ghz_mapping):
        self.vertex_pos_map = {} #vertex -> mode
        self.fusion_map = {} #edge -> modes
        self.ghz_mapping = ghz_mapping
        self.circuit = None
        self.num_modes = 0
        self.initial_vertex_mapping = {}
        self.operation_map = []
    
    def add_ghz_of_vertex(self, vertex: int):
        for ghz in self.ghz_mapping:
            if vertex in ghz:
                self.enlarge_circuit(6)
                for i,v in enumerate(ghz):
                    pos = (self.num_modes-6) + 2*i
                    self.vertex_pos_map[v] = pos
                    self.initial_vertex_mapping[v] = pos
                break
        # print("add ghz of",vertex,"pos map:",self.vertex_pos_map)
    
    def single_sink(self, start_mode, end_mode):
        """only sink single rail instead of dual qubit rail; needed for merge"""
        if start_mode == end_mode:
            return
        for i in range(start_mode, end_mode):
            self.circuit.add((i, i+1), symb.BS.H(np.pi))
        
        for k,v in self.vertex_pos_map.items():
            if v in range(start_mode+1, end_mode):
                self.vertex_pos_map[k] -= 1
            if v == start_mode:
                self.vertex_pos_map[k] = end_mode

    def sink(self, start_mode, end_mode):
        if start_mode == end_mode:
            return
        for i in range(start_mode, end_mode+1):
            self.circuit.add((i, i+1), symb.BS.H(np.pi))

        for i in range(start_mode, end_mode+1):
            self.circuit.add((i, i+1), symb.BS.H(np.pi))
        
        for k,v in self.vertex_pos_map.items():
            if v in range(start_mode+2, end_mode+2):
                self.vertex_pos_map[k] -= 2
            if v == start_mode:
                self.vertex_pos_map[k] = end_mode
        
        print("sink",start_mode,"to",end_mode,"pos map:",self.vertex_pos_map)

    def fuse(self, m1, m2, edge):
        print("fuse",m1,m2,edge)
        assert(m1+2==m2)
        fusion_circ = pcvl.Circuit(4, name="Fuse2")
        fusion_circ.add((0, 1), symb.BS.H())
        fusion_circ.add((2, 3), symb.BS.H())
        fusion_circ.add((1, 2), symb.BS.H(np.pi))
        fusion_circ.add((0, 1), symb.BS.H())
        fusion_circ.add((2, 3), symb.BS.H())
        self.circuit = self.circuit.add(m1, fusion_circ, merge=False)
        edge_key = tuple(sorted(list(edge)))
        if not edge_key in self.fusion_map:
            self.fusion_map[edge_key] = [self.circuit._components[-1]]
        else:
            self.fusion_map[edge_key].append(self.circuit._components[-1])
    
    def enlarge_circuit(self,num_modes):
        self.num_modes += num_modes
        new_circuit = pcvl.Circuit(self.num_modes)
        if self.circuit:
            new_circuit.add(0, self.circuit, merge=True)
        self.circuit = new_circuit
    
    def add_edge(self, edge):
        v1, v2 = edge
        # print("add edge",v1,v2)
        fixed_vertices = self.vertex_pos_map.keys()
        if v1 in fixed_vertices and v2 in fixed_vertices:
            # print("both present")
            v1, v2 = (v1,v2) if self.vertex_pos_map[v1] < self.vertex_pos_map[v2] else (v2,v1)
            self.enlarge_circuit(8)
            
            self.sink(self.vertex_pos_map[v2],self.num_modes-4)
            self.sink(self.vertex_pos_map[v1],self.num_modes-12)

            self.operation_map.append((edge,self.vertex_pos_map[v2],self.num_modes-2))
            self.operation_map.append((edge,self.vertex_pos_map[v1],self.num_modes-10))
            # self.operation_map.append(("sink",self.vertex_pos_map[v2],self.num_modes-4))
            # self.operation_map.append(("sink",self.vertex_pos_map[v1],self.num_modes-12))
            self.vertex_pos_map[v2] = self.num_modes-6
            self.vertex_pos_map[v1] = self.num_modes-8
            self.fuse(self.num_modes-12,self.num_modes-10,edge)
            self.fuse(self.num_modes-4,self.num_modes-2,edge)
            # self.operation_map.append(("fuse",self.num_modes-12,self.num_modes-10))
            # self.operation_map.append(("fuse",self.num_modes-4,self.num_modes-2))
        elif v1 in fixed_vertices or v2 in fixed_vertices:
            #technically an xor
            existing_v, missing_v = (v2,v1) if v2 in fixed_vertices else (v1,v2)
            # print("only",existing_v,"present")
            # possible optimization: depending on whether the vertex is the first or the last in its ghz either first append 4-ghz or first append the 3-vertex ghz.
            self.enlarge_circuit(8)
            self.add_ghz_of_vertex(missing_v)

            self.operation_map.append((edge,self.vertex_pos_map[existing_v],self.num_modes-14))
            self.operation_map.append((edge,self.num_modes - 8,self.vertex_pos_map[missing_v]))

            self.sink(self.num_modes - 8,self.vertex_pos_map[missing_v] - 2)
            self.sink(self.vertex_pos_map[existing_v],self.num_modes-16)
            # self.operation_map.append(("sink",self.num_modes - 8,self.vertex_pos_map[missing_v] - 2))
            # self.operation_map.append(("sink",self.vertex_pos_map[existing_v],self.num_modes-16))
            self.fuse(self.num_modes-16,self.num_modes-14,edge)
            self.fuse(self.vertex_pos_map[missing_v] - 2,self.vertex_pos_map[missing_v],edge)
            # self.operation_map.append(("fuse",self.num_modes-16,self.num_modes-14))
            # self.operation_map.append(("fuse",self.vertex_pos_map[missing_v] - 2,self.vertex_pos_map[missing_v]))
            
            self.vertex_pos_map[existing_v] = self.num_modes-12
            self.vertex_pos_map[missing_v] = self.num_modes-10
        else:
            # print("no vertex present")
            #no vertex present
            self.add_ghz_of_vertex(v1)
            self.enlarge_circuit(8)
            self.add_ghz_of_vertex(v2)
  
            self.operation_map.append((edge,self.vertex_pos_map[v1],self.num_modes-14))
            self.operation_map.append((edge,self.num_modes - 8,self.vertex_pos_map[v2]))
            self.sink(self.vertex_pos_map[v1],self.num_modes-16)
            self.sink(self.num_modes - 8,self.vertex_pos_map[v2] - 2)
            # self.operation_map.append(("sink",self.vertex_pos_map[v1],self.num_modes-16))
            # self.operation_map.append(("sink",self.num_modes - 8,self.vertex_pos_map[v2] - 2))
            self.fuse(self.num_modes-16,self.num_modes-14,edge)
            self.fuse(self.vertex_pos_map[v2] - 2,self.vertex_pos_map[v2],edge)
            # self.operation_map.append(("fuse",self.num_modes-16,self.num_modes-14))
            # self.operation_map.append(("fuse",self.vertex_pos_map[v2] - 2,self.vertex_pos_map[v2]))
            self.vertex_pos_map[v1] = self.num_modes-12
            self.vertex_pos_map[v2] = self.num_modes-10

        # print("vertex positions",self.vertex_pos_map)
    """Wenn man sich einfach nur speichert welche Moden gefust werden sollen, reicht dann nicht die fusion order aus? oder die inverse? """    
    
    def merge2(self, other):
        shared_vertices = set(self.vertex_pos_map.keys()).intersection(other.vertex_pos_map.keys())
        old_size = self.num_modes
        new_size = old_size+other.num_modes-len(shared_vertices)*2
        self.enlarge_circuit(new_size-self.num_modes)
        mode_map = {} #other mode -> new mode
        print("other vertex mapping:",other.initial_vertex_mapping,"self vertex_pos_map",self.vertex_pos_map)
        for vertex,position in other.initial_vertex_mapping.items():
            if not vertex in self.vertex_pos_map.keys():
                self.vertex_pos_map[vertex] = old_size+position-len(shared_vertices)*2
            mode_map[position] = self.vertex_pos_map[vertex]
            mode_map[position+1] = self.vertex_pos_map[vertex]+1
        
        # print(mode_map)
        # fill map other modes-> new modes with remaining 4ghzs
        for i in range(other.num_modes):
            if not i in mode_map.keys():
                next_free_mode = old_size
                while next_free_mode in mode_map.values():
                    next_free_mode += 1
                mode_map[i] = next_free_mode
        
        print("mode map",mode_map)
        print("other operation map",other.operation_map)

        for edge,m1,m2 in other.operation_map:
            pos1, pos2 = (mode_map[m1],mode_map[m2]) if mode_map[m1] < mode_map[m2] else (mode_map[m2],mode_map[m1])
            self.operation_map.append((edge,pos1,pos2))
            self.sink(pos1,pos2-2)
            for k,v in mode_map.items():
                if v in range(pos1+2, pos2):
                    mode_map[k] -= 2
                if v == pos1:
                    mode_map[k] = pos2-2
            self.fuse(pos2-2,pos2,edge)

        # shared_vertices = set(self.vertex_pos_map.keys()).intersection(other.vertex_pos_map.keys())
        # old_size = self.num_modes
        # new_size = old_size+other.num_modes-len(shared_vertices)*2
        # self.enlarge_circuit(new_size-self.num_modes)
        # for vertex,position in other.vertex_pos_map.items():
        #     if not vertex in self.vertex_pos_map.keys():
        #         self.vertex_pos_map[vertex] = old_size+position-len(shared_vertices)*2

        # for i,edge in enumerate(other.fusion_map.keys()):
        #     print("add edge",(edge))
        #     self.add_edge(edge)
        # self.fusion_map = dict(reversed(self.fusion_map.items()))
        # pcvl.pdisplay(self.circuit).save_png("merge2n_ex_it"+str(i)+'.png')
    
    def merge_real(self, other):
        """using component iterator and mode_map
        each time we get a component where modes are not together, 
        we sink the upper near the lower mode and apply the component there
        
        Why components idea does not work (one and for all): 
        If we have a sink going from a new 4-GHZ to a shared vertex, the sinking needs to go from the shared vertex to the 4-GHZ and not vice versa
        It is not possible to just see this on a single component since the sink may start involving only modes in the 4-GHZ cluster already
        We only observe this if the Swaps go from 4-GHZ to the shared vertex.

        What else could we do? Instead of components remeber which modes need to get a sink. 
        But this is tedious. Still not clear why it should parallelize everything as much as possible.
        
        """
        print(self.vertex_pos_map,other.vertex_pos_map)
        shared_vertices = set(self.vertex_pos_map.keys()).intersection(other.vertex_pos_map.keys())
        old_size = self.num_modes
        new_size = old_size+other.num_modes-len(shared_vertices)*2
        self.enlarge_circuit(new_size-self.num_modes)
        mode_map = {} #other mode -> new mode
        for vertex,position in other.initial_vertex_mapping.items():
            if not vertex in self.vertex_pos_map.keys():
                self.vertex_pos_map[vertex] = old_size+position-len(shared_vertices)*2
            mode_map[position] = self.vertex_pos_map[vertex]
            mode_map[position+1] = self.vertex_pos_map[vertex]+1
        
        print(mode_map)
        # fill map other modes-> new modes with remaining 4ghzs
        for i in range(other.num_modes):
            if not i in mode_map.keys():
                next_free_mode = old_size
                while next_free_mode in mode_map.values():
                    next_free_mode += 1
                mode_map[i] = next_free_mode
        
        print(mode_map)
        # import pdb
        # pdb.set_trace()

        for component in other.circuit._components:
            if component[1].name == "CPLX":
                #skip unimportant identity at beginning? Why does it occur anyway?
                continue
            new_modes = [mode_map[m] for m in component[0]]
            print("new modes",new_modes)
            if sorted(new_modes) == list(range(min(new_modes), max(new_modes)+1)):
                # easy case; modes are all together
                self.circuit.add(new_modes,component[1])
            else:
                q1 = new_modes[:len(new_modes)//2]
                q2 = new_modes[len(new_modes)//2:]
                upper, lower = (q1,q2) if q1 < q2 else (q2,q1)
                print("sink",upper,lower)
                for j,m in enumerate(upper):
                    self.single_sink(m, lower[j])
                    for k,v in mode_map.items():
                        if v in range(m+1, lower[j]):
                            mode_map[k] -= 1
                        if v == m:
                            mode_map[k] = lower[j]
                        
                if len(component[0]) > 2:
                    new_modes2 = [lower[0]-i for i in reversed(range(1,len(upper)+1))] + lower
                    try:
                        self.circuit.add(new_modes2, component[1])
                    except:
                        import pdb
                        pdb.set_trace()

    def merge(self, other):
        """what do we know? self and other may share ghzs. Use the position of self
        May they share 4 qubit ghzs? No each 4 qubit ghz is individual per edge"""
        shared_vertices = set(self.vertex_pos_map.keys()).intersection(other.vertex_pos_map.keys())
        print("shared vertices",shared_vertices)
        old_size = self.num_modes
        new_size = old_size+other.num_modes-len(shared_vertices)*2
        # import pdb
        # pdb.set_trace()
        print("new size",new_size)
        self.enlarge_circuit(new_size-self.num_modes)
        for vertex,position in other.vertex_pos_map.items():
            if not vertex in self.vertex_pos_map.keys():
                self.vertex_pos_map[vertex] = old_size+position-len(shared_vertices)*2
        print("new vertex pos map",self.vertex_pos_map)
        shared_modes = [item for v in shared_vertices for item in (other.initial_vertex_mapping[v],other.initial_vertex_mapping[v]+1)]
        shared_modes_map = {} # map other shared modes to self shared modes
        for v in shared_vertices:
            shared_modes_map[other.initial_vertex_mapping[v]] = self.vertex_pos_map[v]
            shared_modes_map[other.initial_vertex_mapping[v]+1] = self.vertex_pos_map[v]+1
        print("shared modes",shared_modes)
        print("shared modes map",shared_modes_map)
        #shared_modes = which modes in other are already present in self.
        for i,component in enumerate(other.circuit._components):
            print("component",component)
            if component[1].name == "CPLX":
                #skip unimportant identity at beginning? Why does it occur anyway?
                continue
            new_modes = []
            for m in component[0]:
                # determine new modes
                if m in shared_modes_map.keys():
                    new_modes.append(shared_modes_map[m])
                else:
                    preceding_shared_modes = [mode for mode in shared_modes if m > mode]
                    mode_subtract = len(preceding_shared_modes)*2
                    new_modes.append(m+old_size-mode_subtract)
            print("new modes",new_modes)
            if sorted(new_modes) == list(range(min(new_modes), max(new_modes)+1)):
                # all modes sequentially in order, just append the gate
                self.circuit.add(new_modes, component[1])
            else:
                #cut needs to be in the half
                upper = new_modes[:len(new_modes)//2]
                lower = new_modes[len(new_modes)//2:]
                for j,m in enumerate(upper):
                    self.single_sink(m, lower[j])
                if len(component[0]) > 2:
                    new_modes2 = [lower[0]-i for i in reversed(range(1,len(upper)+1))] + lower
                    self.circuit.add(new_modes2, component[1])
            #TODO: update shared modes map accordingly
            # if not set(component[0]).intersection(shared_modes):
            #     #no overlap
            #     print("no overlap")
            #     preceding_shared_modes = [mode for mode in shared_modes if component[0][0] > mode]
            #     mode_subtract = len(preceding_shared_modes)*2
            #     new_modes = [m+old_size-mode_subtract for m in component[0]] 
            #     print("new modes",new_modes)
            #     self.circuit.add(new_modes, component[1])
            # elif set(component[0]).intersection(shared_modes) == set(component[0]):
            #     # all modes are shared
            #     print("complete overlap")
            #     new_modes = [shared_modes_map[m] for m in component[0]]
            #     print("new modes",new_modes)
            #     self.circuit.add(new_modes, component[1])
            # else:
            #     #difficult case, some mode overlap
            #     print("partly overlap")
            #     assert(component[0][0] in shared_modes) #hopefully
            #     preceding_shared_modes = [mode for mode in shared_modes if component[0][0] > mode]
            #     mode_subtract = len(preceding_shared_modes)*2
            #     if len(component[0]) > 2:
            #         # this can happen! assume m1,m2 are shared m3,m4 not
            #         end_mode = component[0][2]+old_size-mode_subtract
            #         self.single_sink(shared_modes_map[component[0][0]], end_mode)
            #         edge = None
            #         for k,v in other.fusion_map.items():
            #             if v == (component[0][0],component[0][2]) or v == (component[0][2],component[0][0]):
            #                 edge = k
            #                 break
            #         self.fuse(shared_modes_map[component[0][0]], end_mode, edge)
            #     else:
            #         end_mode = component[0][1]+old_size-mode_subtract
            #         self.single_sink(component[0][0], end_mode)
            pcvl.pdisplay(self.circuit).save_png("merge_ex_it"+str(i)+'.png')  

    def get_fusion_gates_depth(self): 
        #similar to get_mode_count, gets outer loop of fusion gates
        mode_count = [0]*self.circuit.m
        fusion_component_depth = {} #component -> depth
        for component in self.circuit._components:
            if len(component[0]) > 2 and component[1].name != 'Fuse2':
                # skip annoying identity gate at beginning
                continue
            elif component[1].name == 'Fuse2':
                fusion_component_depth[component] = max([mode_count[m] for m in component[0]])
            else:
                mode0, mode1 = component[0]
                outer_loop = max(mode_count[mode0],mode_count[mode1])
                mode_count[mode0] = outer_loop + 1
                mode_count[mode1] = outer_loop
        
        return fusion_component_depth
    
    def get_fusion_order(self):
        component_depth = self.get_fusion_gates_depth()
        edge_order = {}
        for edge,components in self.fusion_map.items():
            max_depth = max([component_depth[c] for c in components])
            max_mode = max(components[0][0] + components[1][0])
            edge_order[edge] = (max_depth, max_mode)
        
        return sorted(edge_order.keys(), key=lambda k: edge_order[k])


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

    

def ghz_mapping_heuristic(G: nx.Graph):
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



def get_node_degrees_map(G):
    node_degrees_map = dict()
    for v in G.nodes:
        n = G.degree[v]
        if not n in node_degrees_map:
            node_degrees_map[n] = [v]
        else:
            node_degrees_map[n].append(v)
    return node_degrees_map

def determine_ghz_mapping_from_fusion_order(fusion_order: List[Tuple[int,int]]):
    initial_ghzs = dict()
    ghz_count = 0
    vertex_to_ghz_map = dict()
    for (v1,v2) in fusion_order:
        if v1 in vertex_to_ghz_map.keys() and v2 in vertex_to_ghz_map.keys():
            continue
        if not v1 in vertex_to_ghz_map.keys() and not v2 in vertex_to_ghz_map.keys():
            ghz_count += 1
            vertex_to_ghz_map[v1] = ghz_count
            vertex_to_ghz_map[v2] = ghz_count
            initial_ghzs[ghz_count] = [v1,v2]
            continue

        existing_v, new_v = (v1,v2) if v1 in vertex_to_ghz_map.keys() else (v2,v1)
        ghz_id = vertex_to_ghz_map[existing_v]
        if len(initial_ghzs[ghz_id]) < 3:
            vertex_to_ghz_map[new_v] = ghz_id
            initial_ghzs[ghz_id].append(new_v)
        else:
            ghz_count += 1
            vertex_to_ghz_map[new_v] = ghz_count
            initial_ghzs[ghz_count] = [new_v]

    return list(initial_ghzs.values())


def build_optimal_tree(tree: Tree, root: TreeNode, lograph: LOGraph2, reversed=False):
    """Build the optimal DFS-ordered circuit for a given tree. The idea is to order each vertex's children, recursively computing the weight of the subtrees. To compute the weight of the subtree we lunch the function on a newly created QTree object.

    :param node: the head of the subtree we are building
    :type node: TreeNode
    :param qtree: the QTree object in which we are building the circuit
    :type qtree: QTree
    :return: the number of outer loops needed to build the subtree with head the TreeNode head. We compute it 
    :rtype: int
    if reversed we get the worst dfs order instead
    """
    # print("in vertex",root.value)
    if not root.children:
        print("no children, return")
        return None, 0
    # print("get children",[child.value for child in root.children])
    children = [(x,build_optimal_tree(tree, x, LOGraph2(lograph.ghz_mapping), reversed)) for x in root.children]
    children.sort(key=lambda x: x[1][1], reverse=reversed)
    # import pdb
    # pdb.set_trace()
    for child,(subgraph,_cost) in children:
        print("add edge",(root.value,child.value))
        lograph.add_edge((root.value,child.value))
        if subgraph and subgraph.circuit:
            try:
                lograph.merge2(subgraph)
            except:
                import pdb
                pdb.set_trace()
        pcvl.pdisplay(lograph.circuit).save_png("merge2n_edge"+str((root.value,child.value))+'.png')

    
    if not lograph.circuit:
        return lograph, 0
    # print("in vertex",root.value,"edge order",edge_order,"children+cost",[(child[0].value,child[1][1]) for child in children])
    # lograph = LOGraph(ghz_mapping, root.value)
    return lograph, lograph.depth()

def get_spanning_trees(graph: nx.Graph, ghz_mapping):
    """removes all edges in ghz mapping from graph and returns spanning trees for remaining graph(s)"""
    sts = []
    gc = graph.copy()
    for ghz in ghz_mapping:
        for i, v in enumerate(ghz):
            if i > 0:
                if (ghz[i-1],v) in gc.edges:
                    gc.remove_edge(ghz[i-1],v)
                    
                else:
                    gc.remove_edge(v,ghz[i-1])
    subgraphs = [gc.subgraph(c).copy() for c in nx.connected_components(gc)]

    for subgraph in subgraphs:
        if subgraph.edges:
            sts.append(get_spanning_tree(subgraph, traverse_method='bfs',depth='max', min_degree=False))

    return sts

#IST DIE FUSION ORDER AUSSAGEKRÄFTIG?? 
"""Also berücksichtigt sie wirklich die Parallelität wie in Francescos Algorithmus?
Fusion order ist eine edge order, aber nicht wirklich die fusion order! TODO: separate methode dafür schreiben. Gemacht, tatsächlich gleichen sie sich im Moment noch
Kann man die spanning tree methode rekursiv machen? Also auf den verbleibenden edges wieder erneut spanning trees generieren?
Warum sollte die Methode jetzt besser sein als beim random-vs-tree-order Vergleich? 
1. Weil die Bäume nicht top down aufgebaut werden
2. Weil man die initialen GHZs mit berücksichtigt

Wie kommt man an die fusion order? 
"""

"""
Welche Vorteile hat der Baum noch, wenn nach dem GHZ Mapping fast alle Edges sowieso teuer sind? 
Wir wollen trotzdem die Fusions so machen, dass sie sich nicht gegenseitig behindern
Aber GHZs sind fix wo können wir dann noch etwas herumschieben? 
Eigentlich ist nicht fix in welcher Reihenfolge die GHZs passieren
Eigentlich könnte man auch die initialen GHZs aus dem Graph entfernen und spanning trees aus den verbleibenden Graphen extrahieren

Was ist der Vorteil bei Francescos Algorithmus für Trees und billige Fusions? 
Critical path. 
GHZs werden dann eingefügt wenn wir sie brauchen, d.h. insbesondere sollte das GHZ Mapping mit allen Qubits nicht unbedingt am Anfang sein.

Two open questions
- Is there an algorithm like dfs search to return the minimal depth circuit for complex edge fusions? Where each edge needs two fusions?
- Is it possible to rewrite the FP calculator to account for complex edge fusions where the two individual fusions don't happen immediately after each other?
+ Maybe the latter is not necessary, we can delay the earlier fusion to happen in the same loop as the later fusion. 
  The construction mechanism so far seems to always put the two fusions in such a way that no other fusion interferes? 
+ Maybe the max outer loop of an individual fusion is also a sufficient metric to determine the fusion order

TODO: Find counterexample where minimal depth of subtrees combined do not yield a minimal depth tree

How should the algorithm look like? 
- At each step: Collect children, add them according to their depth. Not promised to be optimal because vertices are not freely movable (grouped in GHZs?)
- How to keep track of where individual vertices are, vertex_pos_map sufficient? Should be.
"""
