import numpy as np


class Node:

    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Node(data={self.data})"


class KDTree:

    def __init__(self, points):
        self.k = points.shape[1]
        self.root = self._build_tree(points)

    def _build_tree(self, points, depth=0):
        if points.size == 0:
            return None

        axis = depth % self.k

        points = points[np.argsort(points[:, axis])]
        median_idx = points.shape[0] // 2

        points_l = points[:median_idx]
        points_r = points[median_idx+1:]

        return (
            Node(
                data=points[median_idx],
                left=self._build_tree(points_l, depth+1),
                right=self._build_tree(points_r, depth+1)
            )
        )

    def nearest_neighbor(self, point, ignore):
        return self._nearest_neighbor(self.root, point, [None, float("inf")], ignore)

    def _nearest_neighbor(self, node, point, best, ignore, depth=0):
        if node is None:
            return None

        current_point = node.data

        dist = KDTree.distance(point, current_point)
        if dist < best[1] and not any(np.array_equal(current_point, ig) for ig in ignore):
            best[0], best[1] = current_point, dist

        axis = depth % self.k

        if point[axis] <= current_point[axis]:
            next_branch = node.left
            other_branch = node.right
        else:
            next_branch = node.right
            other_branch = node.left

        self._nearest_neighbor(next_branch, point, best, ignore, depth+1)

        # If there could be a closer point on the other side, search it too
        if KDTree.distance(point[axis], current_point[axis]) < best[1]:
            self._nearest_neighbor(other_branch, point, best, ignore, depth+1)

        return best

    @staticmethod
    def _closer_node(point, n1, n2):
        if n1 is None:
            return n2
        elif n2 is None:
            return n1

        d1 = KDTree.distance(point, n1.data)
        d2 = KDTree.distance(point, n2.data)

        if d1 < d2:
            return n1
        else:
            return n2

    @staticmethod
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    def __repr__(self):
        return self.root.__repr__()


if __name__ == "__main__":
    points = np.random.rand(50, 5)
    tree = KDTree(points)
    print(tree)
    point = tree.root.data
    print("point", point)
    print(tree.nearest_neighbor(point, ignore=[point]))
