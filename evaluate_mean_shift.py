import csv
import numpy as np
import matplotlib.pyplot as plt

with open('datasets/2D_data_10000.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    # Iterate over each row in the csv using reader object
    x_coords = []
    y_coords = []
    for row in csv_reader:
        x_coords.append(float(row[0]))
        y_coords.append(float(row[1]))

with open('results/2D_data_3_results.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    # Iterate over each row in the csv using reader object
    x_coords_centr = []
    y_coords_centr = []
    for row in csv_reader:
        x_coords_centr.append(float(row[0]))
        y_coords_centr.append(float(row[1]))



x_coords = np.array((x_coords))
y_coords = np.array((y_coords))
x_coords_centr = np.array((x_coords_centr))
y_coords_centr = np.array((y_coords_centr))

plt.scatter(x_coords, y_coords)
#plt.scatter(x_coords_centr, y_coords_centr)
plt.show()