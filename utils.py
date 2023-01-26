
import networkx as nx
from heapq import heappop, heappush
from itertools import count


def lipschitz(list_a):
    return max([abs(list_a[i] - list_a[i+1]) for i in range(len(list_a)-1)])


def total_variation(list_a):
    return sum([abs(list_a[i] - list_a[i+1]) for i in range(len(list_a)-1)])


def lipschitz_3d(lst):
    # Find the lipschitz constant of a list of 3d points
    return max([abs(lst[i][0] - lst[i+1][0]) + abs(lst[i][1] - lst[i+1][1]) + abs(lst[i][2] - lst[i+1][2]) for i in range(len(lst)-1)])

def total_variation_3d(lst):
    # Find the total variation of a list of 3d points
    return sum([abs(lst[i][0] - lst[i+1][0]) + abs(lst[i][1] - lst[i+1][1]) + abs(lst[i][2] - lst[i+1][2]) for i in range(len(lst)-1)])

# YenKSP: graph, source, target, k -> paths
def YenKSP(g, source, target, K, weight='weight'):

    # copy the graph
    graph = g.copy()

    # Initialize the shortest path
    A = []
    A.append(nx.shortest_path(graph, source, target, weight=weight))

    # Initilize heap to store the potential kth shortest path
    B = []

    for k in range(1, K):

        try:
            # The spur node ranges from the first node to the next to last node in the previous k-shortest path
            for i in range(len(A[k-1]) - 2):

                # Spur node is retrieved from the previous k-shortest path, k − 1
                spur_node = A[k-1][i]

                # The sequence of nodes from the source to the spur node of the previous k-shortest path
                root_path = A[k-1][:i]

                for path in A:
                    if len(path) > i and root_path == path[:i]:
                        try:
                            graph.remove_edge(path[i], path[i+1])
                        except nx.exception.NetworkXError:
                            # break out of all loops
                            raise StopIteration

                for node in root_path:
                    if node != spur_node:
                        graph.remove_node(node)
                
                # Calculate the spur path from the spur node to the target
                try:
                    spur_path = nx.shortest_path(graph, spur_node, target, weight=weight)
                except nx.exception.NetworkXNoPath:
                    raise StopIteration

                # Entire path is made up of the root path and spur path
                total_path = root_path + spur_path

                # Add the potential k-shortest path to the heap, using the length of the path as the priority
                heappush(B, (len(total_path), total_path))

                # Add back the edges and nodes that were removed from the graph
                graph = g.copy()

            if len(B) == 0:
                break
            
            A.append(heappop(B)[1])
        
        except StopIteration:
            break

    return A


# Test code
if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge('a', 'b', weight=7)
    G.add_edge('a', 'c', weight=9)
    G.add_edge('a', 'f', weight=14)
    G.add_edge('b', 'c', weight=10)
    G.add_edge('b', 'd', weight=15)
    G.add_edge('c', 'd', weight=11)
    G.add_edge('c', 'f', weight=2)
    G.add_edge('d', 'e', weight=6)
    G.add_edge('e', 'f', weight=9)

    print(YenKSP(G, 'a', 'e', 3))