#Mean Shift 2D sequential implementation

import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
num_iter = 100
n = 0
bandwidth = 2


def mean_shift(original_points, shifted_points):
    newPosition_x = 0.0
    newPosition_y = 0.0
    tot_weight = 0.0
    for i in range(n):
        difference_x = shifted_points[0] - original_points[i, 0]
        difference_y = shifted_points[1] - original_points[i, 1]
        square_distance = pow(difference_x, 2) + pow(difference_y, 2)
        weight = math.exp((-square_distance)/(2*pow(bandwidth, 2)))
        newPosition_x += original_points[i, 0] * weight
        newPosition_y += original_points[i, 1] * weight
        tot_weight += weight
    newPosition_x /= tot_weight
    newPosition_y /= tot_weight
    shifted_points[0] = newPosition_x
    shifted_points[1] = newPosition_y


#Read the data
with open('datasets/2D_data_3.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    # Iterate over each row in the csv using reader object
    x_coords = []
    y_coords = []
    for row in csv_reader:
        n = n+1
        x_coords.append(float(row[0]))
        y_coords.append(float(row[1]))

points = np.vstack((x_coords, y_coords))
points = points.T
shifted_points = np.copy(points)

for iter in range(num_iter):
    print('Iter: ' + str(iter))
    for i in range(n):
        mean_shift(points, shifted_points[i])

print(np.unique(np.array(shifted_points.round(decimals=2)), axis=0))


