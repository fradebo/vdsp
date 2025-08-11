
import json
import numpy as np
import networkx as nx
import collections
import copy
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from lo_circuit.minimal_swaps import *
import perceval as pcvl

def show(graph_states,idx):
	graph_state = graph_states[idx]
	plt.title("Graph State "+str(idx))
	nx.draw(graph_state,with_labels=True)
	path_name = "Figs of states"
	figs_path = Path(path_name)
	figs_path.mkdir(exist_ok=True)

	# plt.savefig(figs_path / "plot.png")
	# plt.close()
	plt.savefig(path_name+"/"+str(idx)+"w.png")
	plt.close()
	print("Figure saved to \""+path_name+"/"+str(idx)+"w.png\"")

def get_proc_order():
    f = open('proc_order.json')
    proc_order = json.load(f)["order"]
    proc_order = [sorted(el) for el in proc_order]
    return proc_order

def get_ghz_mapping():
    f = open('ghz_mapping.json')
    mapping = json.load(f)["mapping"]
    return mapping

"""
Input: fusion order (not edge order anymore)
requires that we label modes in ascending order (0 are the first two modes aka. the first qubit, 1 are modes 2 and 3 and so on...)
If a fusion fails, we add new modes at the end and prepend fusions to the fusion order such that the graph is rebuilt up to the point where it has been before the fusion
Sometimes it is easier to discard old modes even if they are still present because they are connected to less than 2 other qubits. 
But all we need to make sure when modeling the failures is that after each failure the processing order is adapted accordingly and still generates the desired graph.

Open questions: 
- How do we obtain a mapping of the graph node (outside of the FPT calculator) -> mode? Not necessarily a problem of here, we just want to model the creation process
"""

def get_fp_times(fusion_order, initial_mode_count, fusion_success_probabilities, draw_circuits=False, verbose=False):

    initial_edges = []
    # mode_count = max([v for edge in fusion_order for v in edge]) + 1
    assert(initial_mode_count % 3 == 0) # only ghz inputs for now
    for mode in range(0, initial_mode_count, 3):
        for i in range(0,2):
            initial_edges.append((mode+i, mode+i+1))

    graph_state = nx.Graph(initial_edges)
    numbering = collections.defaultdict()
    idx_to_state = collections.defaultdict()
    mode_count = initial_mode_count
    transitions = []
    cur_idx = 0

    def process_state(state, fusion_order):
        hash = nx.weisfeiler_lehman_graph_hash(state)
        if hash not in numbering:
            dfs(state,fusion_order)
        return numbering[hash][0]
    
    def find_next_fusion(state: nx.Graph, fusion_order):
        for v1,v2 in fusion_order:
            if v1 in state.nodes:
                assert(v2 in state.nodes)
                return v1,v2
        return None, None
    
    def add_new_ghz(state: nx.Graph, mode_count):
        for i in range(mode_count, mode_count+3):
            state.add_node(i)
            if i > mode_count:
                state.add_edge(i-1,i)
        return mode_count + 3
    
    def dfs(state: nx.Graph, current_fusion_order):
        nonlocal verbose
        nonlocal cur_idx
        nonlocal mode_count
        if verbose:
            print("in dfs for graph with nodes",state.nodes,"fusion order",current_fusion_order)
            nx.draw(state,with_labels=True)
            plt.show()

        hash = nx.weisfeiler_lehman_graph_hash(state)
        numbering[hash] = (cur_idx, len(state.nodes))
        idx_to_state[cur_idx] = state
        cur_idx+=1
        idx_orig = numbering[hash][0]

        v1,v2 = find_next_fusion(state,current_fusion_order)
        if v1==None: return
        
        n1,n2 = (list(state.neighbors(v1)),list(state.neighbors(v2)))
        success_graph = copy.deepcopy(state)
        success_graph.remove_nodes_from([v1,v2])
        failure_graph = copy.deepcopy(success_graph)
        #success
        if verbose:
            print("fusion ",(v1,v2)," success")
        for u in n1:
            for v in n2:
                success_graph.add_edge(u,v)
        
        idx_transition = process_state(success_graph, current_fusion_order)
        transitions.append([idx_orig,idx_transition,"p"])

        #failure
        if verbose:
            print("fusion failure ",v1,v2)
        new_fusions = []
        vertex_map = dict() # vertex map should be used to update remaining fusion order to have the correct modes
        orig_neighbors = {v1: n1, v2: n2}
        # print("orig neighbors",orig_neighbors)
        for vertex in [v1,v2]:
            if verbose: print("rebuilding vertex",vertex)
            initialized_ghz = False
            # find initial ghz
            for neighbor in orig_neighbors[vertex]:
                if neighbor in vertex_map:
                    #special case: v1 rebuild has already taken care of this node, skip in v2
                    continue
                connected_components = nx.node_connected_component(failure_graph, neighbor)
                connected_components.discard(set(orig_neighbors[v1 if vertex == v2 else v2])) #prevent interfering neighbors of v1 and v2
                if verbose: print("connected components for neighbor",neighbor,":",connected_components)
                component_without_neighbors = set(connected_components).difference(set(orig_neighbors[vertex])).difference(set([neighbor]))

                if len(connected_components) < 2:
                    # those cases do not require ghzs
                    continue
                elif len(connected_components) == 2: # and  len(component_without_neighbors) == 1:
                    # we found a neighbor who only has a single other neighbor not connected to the vertex, this is exactly mapped with a ghz.
                    second_vertex = set(connected_components).difference(set([neighbor])).pop()
                    vertex_map[vertex] = mode_count + 2
                    vertex_map[neighbor] = mode_count + 1
                    vertex_map[second_vertex] = mode_count
                    mode_count = add_new_ghz(failure_graph, mode_count)
                    failure_graph.remove_nodes_from(connected_components) #cleanup
                    if verbose: print("set initial ghz as ",(vertex,neighbor,second_vertex))
                    if not second_vertex in component_without_neighbors:
                        #special case, second vertex is also a neighbor of vertex, we have to add two more ghzs for the loop fusion
                        new_fusions.append((vertex_map[vertex],mode_count))
                        new_fusions.append((mode_count+2,mode_count+3))
                        new_fusions.append((mode_count+5,vertex_map[second_vertex]))
                        vertex_map[vertex] = mode_count + 1
                        vertex_map[second_vertex] = mode_count + 4
                        mode_count = add_new_ghz(failure_graph, mode_count)
                        mode_count = add_new_ghz(failure_graph, mode_count)
                else:
                    # we found a neighbor with a more complicated subgraph, then we add a ghz where the endpoint is fused with the subgraph
                    new_fusions.append((neighbor,mode_count))
                    vertex_map[vertex] = mode_count + 2
                    vertex_map[neighbor] = mode_count + 1
                    mode_count = add_new_ghz(failure_graph, mode_count)
                    if verbose: print("set initial ghz as ",(vertex,neighbor,mode_count),"fusing the last with neighbor",neighbor)
                initialized_ghz = True
                break
            if not initialized_ghz:
                # no subgraph, then just use two neighbors instead of one for the initial ghz
                # it should never happen that we only have a single orig_neighbor
                if verbose: print("set initial ghz by choosing the first two elements of",orig_neighbors[vertex])
                vertex_map[orig_neighbors[vertex][0]] = mode_count + 2
                vertex_map[vertex] = mode_count + 1
                vertex_map[orig_neighbors[vertex][1]] = mode_count
                mode_count = add_new_ghz(failure_graph, mode_count)
                failure_graph.remove_nodes_from([orig_neighbors[vertex][0],orig_neighbors[vertex][1]]) #cleanup

            # now we add the remaining neighbors:
            for neighbor in orig_neighbors[vertex]:
                if neighbor in vertex_map:
                    # we already added this neighbor
                    continue
                connected_components = nx.node_connected_component(failure_graph, neighbor)
                connected_components.discard(set(orig_neighbors[v1 if vertex == v2 else v2])) #prevent interfering neighbors of v1 and v2
                component_without_neighbors = set(connected_components).difference(set(orig_neighbors[vertex])).difference(set([neighbor]))
                if len(connected_components) < 2:
                    # add single ghz
                    new_fusions.append((vertex_map[vertex],mode_count))
                    vertex_map[vertex] = mode_count + 1
                    vertex_map[neighbor] = mode_count + 2
                    mode_count = add_new_ghz(failure_graph, mode_count)
                    failure_graph.remove_nodes_from(connected_components) #cleanup
                    if verbose: print("add singular vertex",neighbor)
                elif len(connected_components) == 2:
                    second_vertex = set(connected_components).difference(set([neighbor])).pop()
                    new_fusions.append((vertex_map[vertex],mode_count))
                    new_fusions.append((mode_count+2,mode_count+3))
                    vertex_map[vertex] = mode_count + 1
                    vertex_map[neighbor] = mode_count + 4
                    vertex_map[second_vertex] = mode_count + 5
                    mode_count = add_new_ghz(failure_graph, mode_count)
                    mode_count = add_new_ghz(failure_graph, mode_count)
                    failure_graph.remove_nodes_from(connected_components) #cleanup
                    if not second_vertex in component_without_neighbors:
                        #special case, second vertex is also a neighbor of vertex, we have to add two more ghzs for the loop fusion
                        new_fusions.append((vertex_map[vertex],mode_count))
                        new_fusions.append((mode_count+2,mode_count+3))
                        new_fusions.append((mode_count+5,vertex_map[second_vertex]))
                        vertex_map[vertex] = mode_count + 1
                        vertex_map[second_vertex] = mode_count + 4
                        mode_count = add_new_ghz(failure_graph, mode_count)
                        mode_count = add_new_ghz(failure_graph, mode_count)
                else:
                    # connect subgraph, only in this case we do not remove the existing nodes
                    new_fusions.append((vertex_map[vertex],mode_count))
                    new_fusions.append((mode_count+2,mode_count+3))
                    new_fusions.append((mode_count+5,neighbor))
                    vertex_map[vertex] = mode_count + 1
                    vertex_map[neighbor] = mode_count + 4
                    mode_count = add_new_ghz(failure_graph, mode_count)
                    mode_count = add_new_ghz(failure_graph, mode_count)
                    if verbose: print("add subgraph of neighbor",neighbor)
        if verbose: print("vertex map:",vertex_map)
        idx = current_fusion_order.index((v1,v2))
        new_fusion_order = []
        old_fusion_order = current_fusion_order[:]
        for fusion in current_fusion_order[idx:]:
            new_mode0 = vertex_map[fusion[0]] if fusion[0] in vertex_map else fusion[0]
            new_mode1 = vertex_map[fusion[1]] if fusion[1] in vertex_map else fusion[1]
            new_fusion_order.append((new_mode0,new_mode1))
        current_fusion_order = current_fusion_order[:idx-1] + new_fusions + new_fusion_order
        if draw_circuits:
            #determine qubit ordering?
            # import pdb
            #TODO: indices become quite large; can we assume a garbage collector? 
            #Everything else looks good, but where do we get the right indices? 
            # pdb.set_trace()
            old_num_qubits = int(max([qubit for fusion in old_fusion_order for qubit in fusion])/3)*3+3
            num_qubits = int(max([qubit for fusion in current_fusion_order for qubit in fusion])/3)*3+3
            source_order = build_final_qubit_order(old_fusion_order, old_num_qubits)
            target_order = append_to_final_qubit_order(source_order, new_fusions+new_fusion_order, range(old_num_qubits,num_qubits))
            # min_swaps = find_min_swaps(target_order)
            intermediate_source_order = build_final_qubit_order(old_fusion_order, num_qubits)
            import pdb
            pdb.set_trace()
            # print("before",intermediate_source_order)
            # print("after",target_order)
            min_swaps = find_min_swaps_after_failure(intermediate_source_order, target_order)
            pcvl.pdisplay(build_swap_circuit(reversed(min_swaps), num_qubits)).save_svg("failure"+str(v1)+str(v2)+".svg")
            # import pdb
            # pdb.set_trace()

        idx_transition = process_state(failure_graph, current_fusion_order)
        transitions.append([idx_orig,idx_transition,"q"])

    
    dfs(graph_state, fusion_order)
    if verbose:
        print("All configurations:")
        for idx in idx_to_state:
            print(idx,":",idx_to_state[idx])
        print()
        print("Transitions:")
        for el in transitions:
            print(el)
        print()
        for idx_show in range(len(idx_to_state)):
            show(idx_to_state,idx_show)
    """
    Now that the Markov matrix has been generated, the next part calculates the First Passage matrix. 
    """
    

    """
    Markov matrix of transitions
    """
    dim = len(idx_to_state)

    #we start in state with index 0 and want to end up in state with index 
    # (in here the state with the minimum number of vertices, because if so, all ancillas have been fused already)

    min_len = 3*mode_count
    goal_idx = 0
    for idx, num_nodes in numbering.values():
        if num_nodes < min_len:
            goal_idx = idx
            min_len = num_nodes
    # goal_idx = numbering[(0,*sorted([tuple(el) for el in proc_order]))]
    if verbose:
        print("="*40)
        print("Initial State index = ", 0)
        print("Final State index = ", goal_idx)
    mat_markov = np.zeros((dim,dim))

    # file = open("FP_K4.txt","w")
    probability_map = {}
    # for p in np.geomspace(0.001,1.0,40):
    for p in fusion_success_probabilities:
        prob = {
            'p':p,
            'q':1.0-p
            }

        for el in transitions:
            init, final,entry = el
            mat_markov[final][init] = prob[entry]

        """
        regularize matrix: from goal_idx we let the system go anywhere with equal prob. 
        this makes the matrix inversion stable
        """
        for i in range(dim):
            mat_markov[i][goal_idx] = 1.0/float(dim)
        
        
        E = np.full((dim,dim),1.0)
        A = np.linalg.matrix_power(mat_markov,100000000)
        np.set_printoptions(threshold=sys.maxsize)
        Dmat = np.full((dim,dim),0.0) 
        for i in range(dim):
            Dmat[i][i] = 1.0/A[i][i];
        Z = np.linalg.inv(np.identity(dim)-mat_markov+A)
        Z0 = np.diag(np.diag(Z))
        F = np.matmul(Dmat,np.identity(dim)-Z+np.matmul(Z0,E))
        # print("{",p,",",F[goal_idx][0],"}",end=",")
        # print(p,F[goal_idx][0],file=file)
        probability_map[p] = F[goal_idx][0]
#     allData[str(p)].append(F[dim-3][0])
    if verbose: print("----------------------- probabilities:",probability_map)
    return probability_map



# def get_first_passage_times(proc_order, initial_mapping, fusion_sucess_probabilities, verbose=False):
#     '''
#     In this model there is no distinction between complicated and easy edge, 
#     everything is just easy because all components of a difficult edge are reduced to individual 3-GHZ fusions
#     There should be NO graph building logic in the FPT calculator itself!
#     I.e.: Generating a tree of the form 
#     o-o-o-o<8 with mapping 0-1-2 3-4-5 6-7-8 9-10-11 and proc order 2-3 5-6 8-10
#     How does the state look like? nx forest and proc_order? We need the nx graphs to determine neighbors in case of fusion failure.
#     if vertex not present anymore in graph, what do we do? 
#     I.e. we have an initial ghz 0-1-2 and some vertices attached to 1. Now we fuse a vertex with 2 and fail.
#     Now 2 needs to be rebuilt, but we cannot fall back to the initial cluster, since we already have some additional vertices on 1. 
#     So we may have the problem that we need additional resources which were not available at the beginning.
    
#     Solution?: Add another ghz with vertices x-y-2 and prepend 1-x to the proc_order? Lets try...  
#     '''



#     '''
#     graph_state is initialized with the ghz mapping
#     '''

#     initial_edges = []
#     for vertices in initial_mapping:
#         for i in range(1,len(vertices)):
#             initial_edges.append((vertices[i-1], vertices[i]))
#     graph_state = nx.Graph(initial_edges) #[v for vertices in mapping for v in vertices]
    
#     ''' 
#     numbering: maps a graph-state to a unique index
#     idx_to_state: is the corresponding inverse map
#     '''

#     numbering = collections.defaultdict()
#     idx_to_state = collections.defaultdict()


#     transitions = []

#     cur_idx = 0

#     ancilla_vertex_count = max(graph_state.nodes) + 1

#     def process_state(state, proc_order, initial_mapping):
#         hash = nx.weisfeiler_lehman_graph_hash(state)
#         # print("build hash for graph ",state.nodes,state.edges)
#         if hash not in numbering:
#             dfs(state,proc_order, initial_mapping)
#         # print("weird",state.nodes,state.edges,hash)
#         return numbering[hash][0]
    
#     def find_next_fusion(state: nx.Graph, proc_order):
#         for v1,v2 in proc_order:
#             if v1 in state.nodes:
#                 # if not v2 in state.nodes:
#                     # import pdb
#                     # pdb.set_trace()
#                 assert(v2 in state.nodes)
#                 return v1,v2
#         return None, None

#     def dfs(state: nx.Graph, current_proc_order, initial_mapping):
#         print("in dfs for graph with nodes",state.nodes,"proc order",current_proc_order)
#         nx.draw(state,with_labels=True)
#         plt.show()
#         nonlocal cur_idx
#         nonlocal ancilla_vertex_count
#         hash = nx.weisfeiler_lehman_graph_hash(state)
#         # state_orig = [state[0],state[1][:]]
#         numbering[hash] = (cur_idx, len(state.nodes)) # why assign curr_idx here and not before? 
#         idx_to_state[cur_idx] = state
#         cur_idx+=1
#         idx_orig = numbering[hash][0]

#         v1,v2 = find_next_fusion(state,current_proc_order)
#         if v1==None: return

#         n1,n2 = (state.neighbors(v1),state.neighbors(v2))
#         success_graph = copy.deepcopy(state)
#         success_graph.remove_nodes_from([v1,v2])
#         print("fusion ",(v1,v2)," success")
#         for u in n1:
#             for v in n2:
#                 success_graph.add_edge(u,v)
        
#         idx_transition = process_state(success_graph, current_proc_order, initial_mapping)
#         transitions.append([idx_orig,idx_transition,"p"])
        
#         failure_graph = copy.deepcopy(state)
#         failure_graph.remove_nodes_from([v1,v2])
#         print("fusion ",(v1,v2)," fail")
#         initial_ghz_1 = [ghz for ghz in initial_mapping if v1 in ghz]
#         initial_ghz_2 = [ghz for ghz in initial_mapping if v2 in ghz]
#         new_proc_order = current_proc_order[:]
#         for initial_ghz,vertex in [(initial_ghz_1,v1), (initial_ghz_2,v2)]:
#             ghzvertices = set()
#             if not initial_ghz:
#                 neighbors = state.neighbors(vertex)
#                 # We just need a cleanup routine after failure which looks whether there are connected components with 
#                 # at least one non initial vertex and size <= 2. Those get removed and replaced with the original ghz?
                
#                 # if vertex is already additional and not included in the initial mapping, 
#                 # we just abandon the path here. Yet we need to cleanup vertices somewhere...
#                 continue
#             else:
#                 initial_ghz = initial_ghz[0]
#             for ghzvertex in initial_ghz:
#                 if ghzvertex != vertex and ghzvertex in failure_graph.nodes:
#                     ghzvertices.update(nx.node_connected_component(failure_graph, ghzvertex))
#             print("failure check, ghzvertices of ghz",initial_ghz,":",ghzvertices)
#             # import pdb
#             # pdb.set_trace()
#             if len(ghzvertices) <= 2:
#                 print("fusion ",(v1,v2)," fail restart with initial ghz state",initial_ghz)
#                 # restart with initial ghz state
#                 failure_graph.remove_nodes_from(initial_ghz)
#                 for i,v in enumerate(initial_ghz):
#                     failure_graph.add_node(v)
#                     if i > 0:
#                         failure_graph.add_edge(initial_ghz[i-1],v)
#                 # import pdb
#                 # pdb.set_trace()
#             else:
#                 # else we keep the graph and add the missing ghz vertex with a separate fusion
#                 # take care that empty or 2-ary nodes are removed each time, so that the hashes fit.
#                 print("try to rebuild initial ghz state",initial_ghz)
#                 ghz_neighbors = []
#                 complicated = False
#                 if vertex == initial_ghz[0]:
#                     ghz_neighbors.append(initial_ghz[1])
#                 elif vertex == initial_ghz[-1]:
#                     ghz_neighbors.append(initial_ghz[-2])
#                 else:
#                     #This can happen! In our model we do not use n-n fusions, but n-1 and 1-n do happen.
#                     ghz_neighbors.append(initial_ghz[0])
#                     ghz_neighbors.append(initial_ghz[2])
#                     complicated = True
#                     # ghz_neighbors = initial_ghz[0] + initial_ghz[2] 
#                 print("complicated:",complicated)
#                 # if it is the middle vertex we need 3 fusions after all...
#                 # can we really add something to the processing order?? I think so
#                 if not complicated:
#                     nx.relabel_nodes(failure_graph, {ghz_neighbors[0]: ancilla_vertex_count+1}, False) #relabel
#                     ancilla_ghz = [ancilla_vertex_count+2, ghz_neighbors[0], vertex]
#                     for i,v in enumerate(ancilla_ghz):
#                         failure_graph.add_node(v)
#                         if i > 0:
#                             failure_graph.add_edge(ancilla_ghz[i-1],v)
#                     # idx = new_proc_order.index((v1,v2))
#                     new_proc_order = [(ancilla_vertex_count+1, ancilla_vertex_count +2)] + new_proc_order
#                     ancilla_vertex_count += 2
#                     # new_proc_order = new_proc_order[:idx] + [(max_vertex+1, max_vertex +2)] + new_proc_order[idx:]
#                 else:
#                     # we also need to add the second edge which needs three additional fusions
#                     mapping = {ghz_neighbors[0]: ancilla_vertex_count+1, ghz_neighbors[2]: ancilla_vertex_count+8}
#                     nx.relabel_nodes(failure_graph, mapping, False) #relabel
#                     ancilla_ghzs = [[ancilla_vertex_count+2, ghz_neighbors[0], ancilla_vertex_count+3],[ancilla_vertex_count+4, vertex, ancilla_vertex_count+5],[ancilla_vertex_count+6, ghz_neighbors[2], ancilla_vertex_count+7]]
#                     for ancilla_ghz in ancilla_ghzs:
#                         for i,v in enumerate(ancilla_ghz):
#                             failure_graph.add_node(v)
#                             if i > 0:
#                                 failure_graph.add_edge(ancilla_ghz[i-1],v)
#                     # idx = new_proc_order.index((v1,v2))
#                     new_fusions = [(ancilla_vertex_count+1, ancilla_vertex_count +2),(ancilla_vertex_count+3, ancilla_vertex_count +4),(ancilla_vertex_count+5, ancilla_vertex_count +6),(ancilla_vertex_count+7, ancilla_vertex_count +8)]
#                     new_proc_order = new_fusions + new_proc_order
#                     ancilla_vertex_count += 8 
#                     # new_proc_order = new_proc_order[:idx] + new_fusions + new_proc_order[idx:]

#         print("failure graph:",failure_graph.nodes,failure_graph.edges)
#         idx_transition = process_state(failure_graph, new_proc_order, initial_mapping)
#         transitions.append([idx_orig,idx_transition,"q"])        

    
#     dfs(graph_state, proc_order, initial_mapping)
#     if verbose:
#         print("All configurations:")
#         for idx in idx_to_state:
#             print(idx,":",idx_to_state[idx])
#         print()
#         print("Transitions:")
#         for el in transitions:
#             print(el)
#         print()
#         for idx_show in range(len(idx_to_state)):
#             show(idx_to_state,idx_show)
#     """
#     Now that the Markov matrix has been generated, the next part calculates the First Passage matrix. 
#     """
    

#     """
#     Markov matrix of transitions
#     """
#     dim = len(idx_to_state)

#     #we start in state with index 0 and want to end up in state with index 
#     # (in here the state with the minimum number of vertices, because if so, all ancillas have been fused already)

#     min_len = 3*len(initial_mapping)
#     goal_idx = 0
#     for idx, num_nodes in numbering.values():
#         if num_nodes < min_len:
#             goal_idx = idx
#             min_len = num_nodes
#     # goal_idx = numbering[(0,*sorted([tuple(el) for el in proc_order]))]
#     if verbose:
#         print("="*40)
#         print("Initial State index = ", 0)
#         print("Final State index = ", goal_idx)
#     mat_markov = np.zeros((dim,dim))

#     # file = open("FP_K4.txt","w")
#     probability_map = {}
#     # for p in np.geomspace(0.001,1.0,40):
#     for p in fusion_sucess_probabilities:
#         prob = {
#             'p':p,
#             'q':1.0-p
#             }

#         for el in transitions:
#             init, final,entry = el
#             mat_markov[final][init] = prob[entry]

#         """
#         regularize matrix: from goal_idx we let the system go anywhere with equal prob. 
#         this makes the matrix inversion stable
#         """
#         for i in range(dim):
#             mat_markov[i][goal_idx] = 1.0/float(dim)
        
        
#         E = np.full((dim,dim),1.0)
#         A = np.linalg.matrix_power(mat_markov,100000000)
#         np.set_printoptions(threshold=sys.maxsize)
#         Dmat = np.full((dim,dim),0.0) 
#         for i in range(dim):
#             Dmat[i][i] = 1.0/A[i][i];
#         Z = np.linalg.inv(np.identity(dim)-mat_markov+A)
#         Z0 = np.diag(np.diag(Z))
#         F = np.matmul(Dmat,np.identity(dim)-Z+np.matmul(Z0,E))
#         # print("{",p,",",F[goal_idx][0],"}",end=",")
#         # print(p,F[goal_idx][0],file=file)
#         probability_map[p] = F[goal_idx][0]
# #     allData[str(p)].append(F[dim-3][0])
#     return probability_map