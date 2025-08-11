#include <iostream>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <random>

#include <nlohmann/json.hpp>

using namespace std;

void generate_edge_labels(map<pair<int,int>, int> &edge_labels,map<int, pair<int,int>> &label_to_edge, vector<vector<int>> &a){
	int idx = 1;
	for(int i = 0; i < a.size(); i++){
		for(auto j : a[i])
		if(edge_labels[{i,j}]==0){
			edge_labels[{i,j}] = idx;
			edge_labels[{j,i}] = idx;
			label_to_edge[idx] = {i,j};
			idx++;
		}
	}
	return;
}

vector<int> get_state(int n,int nodes){
	vector<int> result(nodes-1);
	for(int i = 0; i < nodes-1; i++){
		result[i] = n%2;
		n /= 2;
	}
	return result;
}

int get_idx(vector<int> state){
	int result = 0;
	for(int i = 0; i < state.size(); i++){
		if(state[i]==1) result += (1<<i);
	}
	return result;
}

int main(int argc, char *argv[]){
	vector<pair<int,int> >target_tree;
	map<pair<int,int>, bool> edge_present;
	map<pair<int,int>, int> edge_labels;
	map<int, pair<int,int>> label_to_edge;
	

	////////////////////////////
	//  Load graph from json  //
	////////////////////////////

	ifstream ifs("inp.json");
	ifstream proc("proc_order.json");
	auto jsonData = nlohmann::json::parse(ifs);
	auto proc_order = nlohmann::json::parse(proc);
	
	int nodes = jsonData.size();
	int edges = nodes - 1;
	// char matrix[3*(1<<(nodes-1))][3*(1<<(nodes-1))];
	const int matrix_dim = 3*(1<<(nodes-1));

	vector<vector< char> >  matrix(matrix_dim, vector<char>(matrix_dim));
	
	vector<vector<int>> a(nodes);

	vector<int> proc_nodes;//order of edge processing
	for (auto& [key, value] : proc_order.items()) {
		assert(value.size() == edges && "number of elements in proc_order.json doesn't match number of edges");
		for(int i = 0; i < value.size(); i++){
			proc_nodes.push_back(value[i]);
		}

	}
	for (auto& [key, value] : jsonData.items()) {
        int intKey = std::stoi(key);  // Convert string key to int
        a[intKey] = vector<int>(value); // Insert into the map
    }

	generate_edge_labels(edge_labels,label_to_edge, a);//run from 1...nodes

	cout<<"Edge labels "<<endl;
	cout<<"============"<<endl;

	for(int i = 1; i <= label_to_edge.size(); i++){
		cout<<i<<" <-> {"<<label_to_edge[i].first<<" "<<label_to_edge[i].second<<"}"<<endl;
	}

	////////////////////
	//  Create Matrix //
	////////////////////

	for(int i = 0; i < matrix_dim; i++){
		for(int j = 0; j < matrix_dim; j++){
			matrix[i][j] = '0';
		}
	}
	for(int i = 0; i < (1<<(nodes-1));i++){//iterate over all edge configurations
		vector<int> state = get_state(i, nodes);
		vector<int> state_copy = state;


		// for(int j = 0; j < state.size(); j++){//look for the first edge j that is absent

		for(int j : proc_nodes){
			if (state[j] == 1) continue;
			state_copy = state;
			
			int node_1 = label_to_edge[j+1].first;
			int node_2 = label_to_edge[j+1].second;
			bool no_neighbors_1 = true;
			bool no_neighbors_2 = true;

			for(auto nn : a[node_1]){
				if(state_copy[edge_labels[{node_1,nn}]-1] == 1){
					no_neighbors_1 = false;
				}
			}
			for(auto nn : a[node_2]){
				if(state_copy[edge_labels[{node_2,nn}]-1] == 1){
					no_neighbors_2 = false;
				}
			}
			if(no_neighbors_1 || no_neighbors_2){//either node_1 or node_2 have no activated edges
				state_copy[j] = 1;
				int idx = get_idx(state_copy);
				state_copy = state;
				//success
				matrix[3*idx][3*i] = 'p';
				matrix[3*i+1][3*i+1] = '1';
				matrix[3*i+2][3*i+2] = '1';
				matrix[3*i+1][matrix_dim-1] = 'x';
				matrix[3*i+2][matrix_dim-1] = 'x';
				matrix[3*i+1][matrix_dim-2] = 'x';
				matrix[3*i+2][matrix_dim-2] = 'x';
				matrix[3*i+1][matrix_dim-3] = 'x';
				matrix[3*i+2][matrix_dim-3] = 'x';


				//failure
				for(auto nn : a[node_1]){
					state_copy[edge_labels[{node_1,nn}]-1] = 0;
				}
				for(auto nn : a[node_2]){
					state_copy[edge_labels[{node_2,nn}]-1] = 0;
				}
				idx = get_idx(state_copy);
				state_copy = state;
				matrix[3*idx][3*i] = 'q';
				break;
			}
			else{
				//success happens in three steps

				state_copy[j] = 1;
				int idx = get_idx(state_copy);
				state_copy = state;

				matrix[3*i+1][3*i] = 'p';
				matrix[3*i+2][3*i+1] = 'p';
				matrix[3*idx][3*i+2] = 'p';

				//in case of failure from state 3*i, only node_1's edges gets reset
				// int node_1 = label_to_edge[j+1].first;
				for(auto nn : a[node_1]){
					// cout<<node_1<<" "<<nn<<" "<<edge_labels[{node_1,nn}]<<endl;
					state_copy[edge_labels[{node_1,nn}]-1] = 0;
				}
				idx = get_idx(state_copy);
				state_copy = state;
				matrix[3*idx][3*i] = 'q';

				//in case of failure from state 3*i+1, we go back to 3*i
				matrix[3*i][3*i+1] = 'q';
				
				//in case of failure from state 3*i+2, only node_2's edges gets reset
				// int node_2 = label_to_edge[j+1].second;
				for(auto nn : a[node_2]){
					state_copy[edge_labels[{node_2,nn}]-1] = 0;
				}
				idx = get_idx(state_copy);
				matrix[3*idx][3*i+2] = 'q';
				break;
			}
		}
	}
	matrix[3*(1<<(nodes-1))-1][3*(1<<(nodes-1))-1] = '1';
	matrix[3*(1<<(nodes-1))-2][3*(1<<(nodes-1))-2] = '1';
	matrix[3*(1<<(nodes-1))-3][3*(1<<(nodes-1))-3] = '1';

	ofstream file;
	file.open("matrix.txt");
	for(int i = 0; i < 3*(1<<(nodes-1)); i++){
		for(int j = 0; j < 3*(1<<(nodes-1)); j++){
			file<<matrix[i][j]<<" ";
		}
		file<<endl;
	}
	
	return 0;
}


//g++ -std=c++17 -I/Users/billyjay/Documents/Programming/libs/JSON/ tree.cpp