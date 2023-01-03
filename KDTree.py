import numpy as np
from queue import PriorityQueue
    

# KDTreeNode: points, depth -> KDTreeNode
class KDTreeNode:
    def __init__(self, points, depth = 0):
        self.points = points
        self.left = None
        self.right = None
        self.mid = None

        self.split_axis = depth % 3

        # sort points by split_axis
        points = points[points[:, self.split_axis].argsort()]

        if len(points) > 1:
            # split points into left and right
            mid = len(points) // 2
            self.mid = points[mid]
            self.left = KDTreeNode(points[:mid], depth + 1)
            self.right = KDTreeNode(points[mid:], depth + 1)


# KDTree: points -> KDTree
class KDTree:
    def __init__(self, points, constrained_axis=None):
        self.constrained_axis = constrained_axis
        self.root = KDTreeNode(points)

    # query: point, k -> (distances, points)
    def query(self, point, k):
        # Create a priority queue
        queue = PriorityQueue(maxsize=k)

        # Recursively search the tree
        self._query(self.root, point, k, queue)
        # Return the k nearest neighbors
        distances, points = np.array(queue.queue).T
        return -distances, np.stack(points)


    # _query: node, point, k, queue -> None
    def _query(self, node, point, k, queue):
        # If the node is empty, return
        if node is None:
            return

        # If the node is a leaf, check the points in the node
        if node.left is None and node.right is None:
            for p in node.points:

                # Custom skip
                if self.constrained_axis is not None:
                    if p[self.constrained_axis] <= point[self.constrained_axis]:
                        continue

                # Compute the distance from the point to the query point
                dist = np.linalg.norm(point - p)

                # If the queue is not full, add the point
                if queue.qsize() < k:
                    queue.put((-dist, p))

                # Otherwise, check if the point is closer than the furthest point in the queue
                else:
                    # If the point is closer, remove the furthest point and add the new point
                    if dist < -queue.queue[0][0]:
                        queue.get()
                        queue.put((-dist, p))

            return
        
        # If the node is not a leaf, check which side of the splitting plane the point is on
        # Compute the distance from the point to the splitting plane
        dist = point[node.split_axis] - node.mid[node.split_axis]

        # Search the side of the splitting plane that is closest to the point
        if dist < 0:
            if node.left is not None:
                self._query(node.left, point, k, queue)
            if node.right is not None:
                self._query(node.right, point, k, queue)
        else:
            if node.right is not None:
                self._query(node.right, point, k, queue)
            if node.left is not None:
                self._query(node.left, point, k, queue)


# Test the KDTree
if __name__ == '__main__':
    # Create a random set of points
    points = np.random.rand(50, 3)

    # Create a KDTree
    tree = KDTree(points)

    # Query the tree
    k = 5
    point = np.random.rand(3)
    distances, neighbors = tree.query(point, k)


    # Visualize the results
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw the k nearest neighbors in red
    ax.scatter(neighbors[:, 0], neighbors[:, 1], neighbors[:, 2], c='r', marker='o')


    # Draw original points (except the nearest neighbors) in gray
    mask = np.ones(len(points), dtype=bool)

    # Get indices of nearest neighbors
    indices = np.array([np.where((points == n).all(axis=1))[0][0] for n in neighbors])

    # Set mask to False for nearest neighbors
    mask[indices] = False

    # Draw the remaining points in gray
    ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], c='gray', marker='o')

    # Draw the query point in blue 
    ax.scatter(point[0], point[1], point[2], c='b', marker='o')
    plt.show()
