import argparse
import pandas as pd 
from collections import defaultdict
from typing import DefaultDict, Dict, Set, TextIO, Tuple, Union



class Graph:
    _nodes = set()  # Set of nodes
    _in_degree = dict()  # key: node, value: in-degree of node
    _out_degree = dict()  # key: node, value: out-degree of node
    _node_info = dict() 

    @property
    def nodes(self) -> Set[int]:
        return self._nodes

    @property
    def in_degree(self) -> Dict[int, int]:
        return self._in_degree

    @property
    def out_degree(self) -> Dict[int, int]:
        return self._out_degree
    
    @property
    def node_info(self) -> Dict[int, str]:
        return self._node_info



class Graph(Graph):
    def __init__(self,edge_class, filePath: str) -> None:
    
        self._nodes = set()
        

        for i in range(572):
            self._nodes.add(i)
            
        max_v  = max(list(self._nodes))

        in_degree = {key: 0 for key in range(max_v+1)}
        out_degree = {key: 0 for key in range(max_v+1)}
        out_neighbor = {key: set([]) for key in range(max_v+1)}
        
       
  
        with open(filePath + ".txt", "r") as f:
            f.readline()  # Skip header line
            while True:
                line = f.readline()
                if not line:
                    break
                line_split = line.rstrip().split(" ")
                if int(line_split[0]) == edge_class:
                    
                    src = int(line_split[1])
                    dst = int(line_split[2])
                    in_degree[dst] += 1 
                    out_degree[src] += 1 
                    out_neighbor[src].add(dst)


        self._in_degree = dict(in_degree)
        self._out_degree = dict(out_degree)
        self.__out_neighbor: Dict[int, Set[int]] = dict(out_neighbor)

    

                   
    @property
    def out_neighbor(self) -> Dict[int, Set[int]]:
        
        return self.__out_neighbor




def preference_degree(graph) -> Dict[int, float]:
 

    preference: Dict[int, float] = dict()
    total = 0
    
    for i in graph.in_degree.values():
        total += i+1 

    
    preference = {n: (graph.in_degree[n]+1)/total for n in graph.in_degree}
    return preference

def l1_distance(x: DefaultDict[int, float], y: DefaultDict[int, float]) -> float:
    err: float = 0.0
    for k in x.keys():
        err += abs(x[k] - y[k])
    return err



def pagerank_v(graph,damping_factor,preference,maxiters,tol):
    vec: DefaultDict[int, float] = defaultdict(float)  # Pagerank vector
    max = len(graph.nodes)
    vec = {node: 1/max for node in graph.nodes}
    
    out_neighbor = graph.out_neighbor
    for itr in range(maxiters):
        new_vec = defaultdict(float)

        for node, page_rank in vec.items():

            out = graph.out_degree[node]

            for j in list(out_neighbor[node]):
            
                new_vec[j] += (damping_factor * (page_rank/out))
            new_vec[node] += ((1-damping_factor) * preference[node])
            

        delta: float = 0.0

        delta = l1_distance(new_vec,vec)
        print(f"[Iter {itr}]\tDelta = {delta}")
    
        vec = new_vec

        
        if delta < tol:
            
            break
        
       
    return dict(vec)

graphs = []
pagerank = []

### edge type 65개 별로 생성된 65개의 graph들에 맞는 edge의 message passing pagerank 계산
for i in range(65):
    fileName = ""

    beta = 0.85  # damping factor
    maxiters = 1000
    tolerance = 1e-6
    graph_memory = Graph(i,"../data/full_pos2")
    
    

    preference: Dict[int, float]

    preference = preference_degree(graph_memory)
    vec = pagerank_v(graph_memory, beta, preference, maxiters, tolerance)
    graphs.append(graph_memory)
    pagerank.append(vec)
    
    

input_file = '../data/full_pos2.txt'
output_file = '../data/full_pos2_weight.csv'


df = pd.read_csv(input_file,sep = ' ')


def calculate_weight(row):
    
    label = row['label']
    src = row['src']
    dst = row['des']
    
    out_number = pagerank[label][src]/ len(graphs[label].out_neighbor[src])
    
    
    return out_number

# 각 행에 대해 weight 값 계산
df['weight'] = df.apply(calculate_weight, axis=1)

# page_rank weight 파일 생성 
df.to_csv(output_file, index=False)
