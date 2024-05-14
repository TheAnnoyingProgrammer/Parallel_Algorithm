import time
import networkx as nx
import matplotlib.pyplot as plt
from mpi4py import MPI

def SCC(iGraph):
    CCTable = dict()
    n = len(iGraph) - 1
    for node in iGraph.nodes:
        DTable = nx.shortest_path_length(iGraph, source=node)
        CCTable[node] = n / sum(value for value in DTable.values())
    return CCTable

def PCC(iGraph, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    WList = list(iGraph.nodes)[rank::size]
    PCCTable = dict()
    n = len(iGraph) - 1
    for node in WList:
        DTable = nx.shortest_path_length(iGraph, source=node)
        PCCTable[node] = n / sum(value for value in DTable.values())
    CCTable = comm.gather(PCCTable, root=0)
    return CCTable

if __name__ == '__main__':
    GraphOne = nx.Graph()

    with open('EdgeList.txt', 'r') as f:
        for line in f:
            edge = line.strip().split()
            GraphOne.add_edge(edge[0], edge[1])

    ts = time.time()
    CCTableOne = SCC(GraphOne)
    tf = time.time()
    print("SCC Table:", CCTableOne)
    print("Time taken for SCC:", tf - ts)

    ts = time.time()
    comm = MPI.COMM_WORLD
    CCTableTwo = PCC(GraphOne, comm)
    tf = time.time()
    if comm.rank == 0:
        combined_centrality = {}
        for cent_dict in CCTableTwo:
            combined_centrality.update(cent_dict)
        print("Closeness Centrality Table:", combined_centrality)
        print("Time taken for closeness centrality:", tf - ts)

    nx.draw(GraphOne, with_labels=True)
    plt.show()
