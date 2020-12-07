import numpy as np

class GraphMST:
    def __init__(self, graph):
        self.num_vertices = graph.shape[0]
        self.graph = graph

    def find_min(self, keys, inMST):
        min = np.inf
        min_idx = 0
        for i in range(self.num_vertices):
            if keys[i] < min:
                min_idx = i
                min = keys[i]
        return min_idx


    def prim(self, root):
        keys = np.ones(self.num_vertices) * np.inf
        keys[root] = 0

        parents = np.zeros(self.num_vertices, dtype = 'int')
        parents[root] = -1
        inMST = np.zeros(self.num_vertices)

        MST = np.zeros((self.num_vertices, self.num_vertices))
        MST_edges = np.zeros((self.num_vertices, 2))

        for i in range(self.num_vertices):
            picked = keys.argmin()
            for j in range(self.num_vertices):
                if inMST[j] == 0 and self.graph[picked, j] > -np.inf and keys[j] > self.graph[picked, j]:
                    keys[j] = self.graph[picked, j]
                    parents[j] = picked

            MST[parents[picked], picked] = 1

            MST_edges[i, 0] = parents[picked]
            MST_edges[i, 1] = picked

            if picked != root:
                MST[picked, parents[picked]] = 1
            keys[picked] = np.inf
            inMST[picked] = 1


        return MST, MST_edges[1:], parents



