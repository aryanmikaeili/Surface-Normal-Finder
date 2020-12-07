
import numpy as np
import plotly.graph_objects as go
from Graph_mst import GraphMST


class node:
    def __init__(self, label):
        self.label = label
        self.children = []


def find_distance(x, y):

  ##################################Inputs##################################
  # x : (N1 * f) ndarray
  # y : (N2 * f) ndarray
  ##########################################################################

  ##################################Outputs##################################
  #o:  (N1 * N2) ndarray , o[i, j] is the distance of x[i] and y[j] squared
  ##########################################################################

  k = y.shape[0]
  data_size = x.shape[0]

  x_norm = np.repeat(np.expand_dims(np.linalg.norm(x, axis = 1) ** 2, 1), k, axis = 1)
  y_norm = np.repeat(np.expand_dims(np.linalg.norm(y, axis = 1) ** 2, 0), data_size, axis = 0)
  x_y_inner = np.matmul(x, y.T)

  o = x_norm + y_norm - 2 * x_y_inner
  return o


def complete_r_graph(k_neighbors, graph):
    num_points = graph.shape[0]
    r = np.arange(0, num_points, 1).reshape(-1, 1)

    graph[r, k_neighbors] = 1
    graph[k_neighbors, r] = 1

    graph[graph == 0] = -np.inf

    return graph

def assign_weights(graph, normals):
    normal_dot = np.matmul(normals, normals.T)
    graph[graph == 1] = 1 - np.abs(normal_dot[graph == 1])
    graph = np.round(graph, 7)
    return graph

def build_tree(mst_parents, num_points):
    nodes = [node(i) for i in range(num_points)]
    for i in range(num_points):
        if mst_parents[i] != -1:
            nodes[mst_parents[i]].children.append(nodes[i])
    return nodes
def propagate_orientations(normals, mst, root):
    num_points  =normals.shape[0]

    mst_nodes = build_tree(mst, num_points)

    if normals[root][2] < 0:
        normals[root] *= -1
    stack = [mst_nodes[root]]
    visited = np.zeros(num_points)
    j = 0
    while len(stack) > 0:
        current = stack.pop(-1)
        if visited[current.label] == 1:
            continue

        if current.label != root and np.inner(normals[mst[current.label]], normals[current.label]) < 0:
            normals[current.label] *= -1

        visited[current.label] = 1
        stack += current.children
        j += 1

    return normals

class Normal_finder:
    def __init__(self, points, k = 20, l = 200):
        self.points = points

        self.point_num = points.shape[0]
        self.k = k
        self.l = l

    def find_k_closest(self, points):
        distances = find_distance(points, points)
        distances = np.round(distances, 7)
        np.fill_diagonal(distances, 0)
        sorted_distances = np.argsort(distances, axis=1)


        k_closest = sorted_distances[:, 1:self.k + 1]
        return points[k_closest], k_closest, distances

    def calculate_covariance(self, matrix):
        zero_mean_mat = matrix - matrix.mean(axis = 1).reshape(-1, 1, 3)
        covariance = np.einsum("nij,njk -> nik ", np.transpose(zero_mean_mat, (0, 2, 1)), zero_mean_mat)
        return covariance / (self.k - 1)

    def find_angles(self, k_args, normals):
        k_closest_normals = normals[k_args]

        angles = np.arccos(np.clip(np.einsum("nji, ni -> nj", k_closest_normals, normals), a_min= -1, a_max= 1)) * 180 / np.pi

        angles_mean = angles.mean(axis = 1)

        return angles_mean

    def find_normals_curvature(self):
        k_closests, k_closests_args, dists = self.find_k_closest(self.points)
        dists[dists == 0] = -np.inf

        covariances = self.calculate_covariance(k_closests)

        eig_values, eig_vectors = np.linalg.eig(covariances)

        eig_values_min_idx = np.argsort(eig_values, axis=1)[:, 0]
        mask = np.eye(3)[eig_values_min_idx].astype('bool')
        
        eig_values_min = eig_values[mask]
        curvature = eig_values_min / eig_values.sum(axis = 1)
        if np.allclose(curvature, 0):
            return None, None, None, None


        normals = eig_vectors.transpose((0,2,1))[mask]


        g = GraphMST(dists)
        tree, _, _ = g.prim(0)

        graph = complete_r_graph(k_closests_args, tree)
        graph = assign_weights(graph, normals)

        g = GraphMST(graph)

        root = self.points.argmax(axis=0)[2]
        _, mst_edges, parents = g.prim(root)

        oriented_normals = propagate_orientations(normals, parents, root)

        angles_mean = self.find_angles(k_closests_args, oriented_normals)

        self.l = angles_mean.mean() / curvature.mean()
        features = angles_mean +  self.l * curvature

        return oriented_normals, curvature, angles_mean, features



def plot_normals(points, normals):
    x, y, z = np.array(points).T
    trace =  go.Scatter3d(x = x, y = y, z = z,  mode = 'markers', marker=dict(color = 'blue', opacity = 0.5, size = 6))
    traces = [trace]
    end_points = points + normals
    for vec in zip(points, end_points):
        x, y, z = np.array(vec).T
        trace =  go.Scatter3d(x = x, y = y, z = z,  mode = 'lines', line = dict(width = 1.5, color = 'red'))
        traces.append(trace)

    return traces

def plot_mst(points, edges):
    x, y, z = np.array(points).T
    trace =  go.Scatter3d(x = x, y = y, z = z,  mode = 'markers', marker=dict(color = 'blue', opacity = 0.5, size = 6))
    traces = [trace]

    for edge in edges:
        st = points[edge[0].astype('int')]
        en = points[edge[1].astype('int')]
        x, y, z = np.array([en,st]).T

        trace =  go.Scatter3d(x = x, y = y, z = z,  mode = 'lines', line = dict(width = 1.5, color = 'red'))
        traces.append(trace)

    return traces


def find_important_points(points, k):
    Nf = Normal_finder(points, k)

    _, _, _, features = Nf.find_normals_curvature()
    if features is None:
        return None
    features_sorted = features.argsort()[::-1]
    return features_sorted


