
import numpy as np

from normal_finder import find_important_points

import plotly
import plotly.graph_objects as go


BASE_DIR = '/Users/aryanmikaeili/PycharmProjects/adversarial/'

data_path = BASE_DIR + 'modelnet40_test_data.npy'

important_points_path = BASE_DIR + 'test_important_points'
unimportant_points_path = BASE_DIR + 'test_unimportant_points'
test_important_idx_path =  BASE_DIR + 'test_important_idx'

checkpoint = BASE_DIR + 'checkpoint'

num_points = 2048
num_keep = 1024

DATA = np.load(data_path)
num_pointclouds = DATA.shape[0]

"""
DATA = np.load(data_path)
important_points = np.load(dest_path + '.npy')

num_pointcloud = DATA.shape[0]
unimportant_points = np.zeros((num_pointcloud, num_points - num_keep, 3))
for i in range(num_pointcloud):
    data = DATA[i]
    i_points = important_points[i]
    idxs = []
    if i == 283:
        a = 0
    for p in i_points:
        found = np.where((data == p).all(axis=1))[0][0]
        idxs.append(found)
    if len(idxs) != 1024:
        Exception("you fucked up something")
    a = 0
    unimportant_points[i] = data[np.delete(np.arange(2048), idxs)]
    print(i)


np.save(BASE_DIR + "unimportant_points", unimportant_points)

"""



idx = np.zeros((num_pointclouds, num_points), dtype='int')

flat_pc = []

for i in range(num_pointclouds):
    current_pointcloud = DATA[i]
    points_sorted = find_important_points(current_pointcloud, 20)
    if points_sorted is None:
        flat_pc.append(i)


        idx[i] = np.arange(0,2048)

        continue

    idx[i] = points_sorted
    print(i, flat_pc)
    if i % 100 == 99:
        np.save(test_important_idx_path, idx)


np.save(test_important_idx_path, idx)


"""
modified = np.load(dest_path + '.npy')
m = np.load(dest_path + '.npy')

for i in range(9840):
    m[i] -= np.mean(m[i], axis = 0)
    n = np.max(np.linalg.norm(m[i], axis = 1))
    if n == 0:
        modified[i] = DATA[i][np.random.choice(num_points, num_keep, replace = False)]
        print(i)


np.save(BASE_DIR + 'debugged_modified_train_data', modified)

"""


