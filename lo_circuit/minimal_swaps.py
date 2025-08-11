from typing import List, Tuple
import perceval as pcvl
import perceval.components as symb
import numpy as np

# def build_final_qubit_order(fusion_order: List[Tuple[int,int]], num_qubits):
#     """idea: graph state qubit is adjacent to at least one fusion qubit in the final qubit_oder.
#     Thus, we first try to prepend all graph state qubits to the respective fusion qubits and then append the remaining"""
#     fusion_qubits = [x for f in fusion_order for x in f]
#     graph_state_qubits = set([x for x in range(num_qubits) if not x in fusion_qubits])
#     final_order = []
#     for fusion_qubit in fusion_qubits:
#         if fusion_qubit-1 in graph_state_qubits:
#             final_order.append(fusion_qubit-1)
#             graph_state_qubits.discard(fusion_qubit-1)
#         final_order.append(fusion_qubit)
#     for last_qubit in sorted(list(graph_state_qubits)):
#         final_order.append(last_qubit) #the set should be almost empty here already
#     return final_order

def build_final_qubit_order(fusion_order: List[Tuple[int,int]], num_qubits):
    """Idea: we fill up graph state vertices so that they need the least amount of swaps?"""
    fusion_qubits = [x for f in fusion_order for x in f]
    graph_state_qubits = sorted([x for x in range(num_qubits) if not x in fusion_qubits])
    final_order = []
    for (q1,q2) in fusion_order:
        maxq = q2
        drop = 0
        for gs_qubit in graph_state_qubits:
            if gs_qubit < maxq:
                drop += 1
                final_order.append(gs_qubit)
        graph_state_qubits = graph_state_qubits[drop:]
        
        final_order.append(q1)
        final_order.append(q2)
    if graph_state_qubits:
        final_order += graph_state_qubits
    return final_order

def append_to_final_qubit_order(source_order, new_fusions, new_qubits):
    final_order = source_order[:]
    fusion_qubits = [x for f in new_fusions for x in f]
    graph_state_qubits = sorted([x for x in new_qubits if not x in fusion_qubits])
    for fusion_qubit in fusion_qubits:
        if not fusion_qubit in new_qubits:
            del final_order[final_order.index(fusion_qubit)]
    
    for (q1,q2) in new_fusions:
        maxq = q2
        drop = 0
        for gs_qubit in graph_state_qubits:
            if gs_qubit < maxq:
                drop += 1
                final_order.append(gs_qubit)
        graph_state_qubits = graph_state_qubits[drop:]
        
        final_order.append(q1)
        final_order.append(q2)
    if graph_state_qubits:
        final_order += graph_state_qubits

    return final_order

def find_min_swaps(target_order: List):
    """min number of adjacent swaps is equal to number of list inversions:
    https://stackoverflow.com/questions/20990127/sorting-a-sequence-by-swapping-adjacent-elements-using-minimum-swaps
    The algorithm here may be suboptimal in runtime, but we get a minimal sequence of swaps
    by repeatedly going through the list and swap any two pairs where the first element is greater than the second.
    target_order = order to which an initial ascending order 1...n should be sorted
    """
    swaps = []
    current_order = target_order.copy()
    while current_order != sorted(current_order):
        for i,x in enumerate(current_order):
            if i > 0 and x < current_order[i-1]:
                swaps.append((i-1,i))
                current_order[i] = current_order[i-1]
                current_order[i-1] = x
    return swaps

def build_swap_circuit(swaps, num_qubits):
    """one swap consists of four beamsplitters on the four modes."""
    c = pcvl.Circuit(num_qubits*2)
    for (p1, p2) in swaps:
        c.add((p1*2+1, p2*2), symb.BS.H(np.pi))
        c.add((p1*2, p1*2+1), symb.BS.H(np.pi))
        c.add((p2*2, p2*2+1), symb.BS.H(np.pi))
        c.add((p1*2+1, p2*2), symb.BS.H(np.pi))
    return c

def find_min_swaps_after_failure(source_order: List, target_order: List):
    #Here we should implement a function which takes a non ascending order as input sorting
    # qubit_map = {}
    # for i,qubit in enumerate(source_order):
    #     qubit_map[qubit] = i
    # adjusted_target_order = [qubit_map[qubit] for qubit in target_order]
    source_map = {v: pos for pos,v in enumerate(source_order)}
    adjusted_target_order = [source_map[v] for v in target_order]
    # import pdb
    # pdb.set_trace()
    return find_min_swaps(adjusted_target_order)