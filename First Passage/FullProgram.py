import string
import numpy as np
import sys
from numpy.linalg import matrix_power
import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx
import collections
from pathlib import Path






def get_adj():
	f = open('inp.json')
	adj = json.load(f)
	target_adj = {int(key):val for key,val in adj.items()}
	for el in target_adj:
		target_adj[el] = sorted(target_adj[el])
	return target_adj

def get_proc_order():
	f = open('proc_order.json')
	proc_order = json.load(f)["order"]
	proc_order = [sorted(el) for el in proc_order]
	return proc_order
	
def show(graph_states,adj,proc_order,idx):
	graph_state = graph_states[idx]
	G = nx.Graph()
	if graph_state[0]==0:
		vertices = [el[0] for el in adj.values()]+[el[1] for el in adj.values()]
		for i in vertices:
			G.add_node(i)
		for edge in graph_state[1]:
			G.add_edge(edge[0],edge[1])
	if graph_state[0]==1:
		vertices = [el[0] for el in adj.values()]+[el[1] for el in adj.values()]
		for i in vertices:
			G.add_node(i)
		G.add_node(-1)
		G.add_node(-2)
		next_edge = find_next_edge(graph_state,proc_order)
		v1,v2 = sorted(proc_order[next_edge])
		for edge in graph_state[1]:
			G.add_edge(edge[0],edge[1])
		G.add_edge(v1,-1)
		G.add_edge(-1,-2)
	plt.title("Graph State "+str(idx))
	nx.draw(G,with_labels=True)
	path_name = "Figs of states"
	figs_path = Path(path_name)
	figs_path.mkdir(exist_ok=True)

	# plt.savefig(figs_path / "plot.png")
	# plt.close()
	plt.savefig(path_name+"/"+str(idx)+".png")
	plt.close()
	print("Figure saved to \""+path_name+"/"+str(idx)+".png\"")

def find_next_edge(graph_state,proc_order):
	'''
	finds the absent edge of graph_state that has to be added next according to proc_order
	'''
	for idx, el in enumerate(proc_order):
		l = sorted([el[0],el[1]])
		if (l[0],l[1]) not in graph_state[1]: return idx
	return -1

def vertex_without_edge(graph_state,v):

	for el in graph_state[1]:
		if v in el: return False
	return True



def delete_edge(state, v1,v2):
	'''
	delete all edges that involve v1 or v2
	'''
	l = [el for el in state[1] if v1 not in el and v2 not in el]
	return [state[0],l]

def blow_up(state, vert):
	'''
	delete all edges that are attached to vertex v1
	'''
	l = [el for el in state[1] if vert not in el]
	return [0, l]


if __name__ == "__main__":
	target_adj = get_adj()
	proc_order = get_proc_order()
	V = len(target_adj) #number of vertices

	'''
	each vertex can be in one of two states:
	 0: o
	 1: o-o-o
	initially all vertices are in state 0; state 1 is produced when we attach a 4-photon GHZ to a vertex, see Fig. 2 of paper
	'''



	'''
	Graphstate is a list. The first entry contains the state of the vertex that is currently processed.
	The second is list that contains the already added edges. All edges [v1,v2], throughout this program, must respect v1<v2.
	Keep the list of vertices, graph_state[1], in sorted form. This keeps us from over-counting identical states.
	'''
	graph_state = [0,[]]
	
	''' 
	numbering: maps a graph-state to a unique index
	idx_to_state: is the corresponding inverse map
	'''

	numbering = collections.defaultdict()
	idx_to_state = collections.defaultdict()


	transitions = []

	cur_idx = 0
	def dfs(state):
		global cur_idx
		numbering[(state[0],*state[1])] = cur_idx
		idx_to_state[cur_idx] = [state[0],state[1][:]]
		cur_idx+=1
		idx_orig = numbering[(state[0],*state[1])]
		next_edge = find_next_edge(state,proc_order)
		if next_edge==-1: return

		state_orig = [state[0],state[1][:]]
		v1,v2 = sorted(proc_order[next_edge])
		if state[0]==0:			
			if vertex_without_edge(state,v1) or vertex_without_edge(state,v2):

				#success:
				state[1].append(tuple(sorted([v1,v2])))
				state[1] = sorted(state[1])

				if (state[0],*state[1]) not in numbering:
					dfs([state[0],state[1][:]])

				idx_transition = numbering[(state[0],*state[1])]
				transitions.append([idx_orig,idx_transition,"p"])
				
				#restore
				state = [state_orig[0],state_orig[1][:]]

				#failure:
				state = delete_edge(state, v1,v2)
				if (state[0],*state[1]) not in numbering:
					dfs([state[0],state[1][:]])

				idx_transition = numbering[(state[0],*state[1])]
				transitions.append([idx_orig,idx_transition,"q"])
			
			else:
				'''
				both v1 and v2 have edges attached, so we have to add the 4-photon GHZ to either v1 or v2 
				(it doesn't matter which we pick, because in the next step we have to do fusion with the other vertex. So both
				vertices will participate in fusion with potential failure.), We will attach it to v1.
				'''

				#success
				state = [1,state_orig[1][:]]

				if (state[0],*state[1]) not in numbering:
					dfs([state[0],state[1][:]])

				idx_transition = numbering[(state[0],*state[1])]
				transitions.append([idx_orig,idx_transition,"p"])

				#restore
				state = [state_orig[0],state_orig[1][:]]

				# failure:
				state = blow_up(state, v1)

				if (state[0],*state[1]) not in numbering:
					dfs([state[0],state[1][:]])

				idx_transition = numbering[(state[0],*state[1])]
				transitions.append([idx_orig,idx_transition,"q"])


		else:
			'''
			here we try to fuse the vertex v2 and the added edges (starting with the 4-GHZ).
			in case of success: we end up adding edge (v1,v2) and setting state[0]=0
			in case of failure: we blow up everything connected to v2 and set state[0]=0
			'''

			#success
			state[1].append(tuple(sorted([v1,v2])))
			state[1] = sorted(state[1])
			state[0] = 0

			if (state[0],*state[1]) not in numbering:
				dfs([state[0],state[1][:]])

			idx_transition = numbering[(state[0],*state[1])]
			transitions.append([idx_orig,idx_transition,"p"])

			#restore
			state = [state_orig[0],state_orig[1][:]]

			# failure:
			state = blow_up(state, v2)
			if (state[0],*state[1]) not in numbering:
				dfs([state[0],state[1][:]])

			idx_transition = numbering[(state[0],*state[1])]
			transitions.append([idx_orig,idx_transition,"q"])

	
	dfs(graph_state)

	print("All configurations:")
	for idx in idx_to_state:
		print(idx,":",idx_to_state[idx])
	print()
	print("Transitions:")
	for el in transitions:
		print(el)
	print()
	for idx_show in range(len(idx_to_state)):
		show(idx_to_state,target_adj,proc_order,idx_show)
	"""
	Now that the Markov matrix has been generated, the next part calculates the First Passage matrix. 
	"""
	

	"""
	Markov matrix of transitions
	"""
	dim = len(idx_to_state)

	#we start in state with index 0 and want to end up in state with index
	goal_idx = numbering[(0,*sorted([tuple(el) for el in proc_order]))]
	print("="*40)
	print("Initial State index = ", 0)
	print("Final State index = ", goal_idx)
	mat_markov = np.zeros((dim,dim))

	file = open("FP_K4.txt","w")
	# for p in np.geomspace(0.001,1.0,40):
	for p in np.linspace(0.001,1.0,40):
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
		

		# print(mat_markov)
		l = [mat_markov[i][3] for i in range(dim)]
		# print(l)
		
		E = np.full((dim,dim),1.0)
		A = matrix_power(mat_markov,100000000)
		np.set_printoptions(threshold=sys.maxsize)
		Dmat = np.full((dim,dim),0.0) 
		for i in range(dim):
			Dmat[i][i] = 1.0/A[i][i];
		Z = np.linalg.inv(np.identity(dim)-mat_markov+A)
		Z0 = np.diag(np.diag(Z))
		F = np.matmul(Dmat,np.identity(dim)-Z+np.matmul(Z0,E))
		# print("{",p,",",F[goal_idx][0],"}",end=",")
		print(p,F[goal_idx][0],file=file)
# 	allData[str(p)].append(F[dim-3][0])
