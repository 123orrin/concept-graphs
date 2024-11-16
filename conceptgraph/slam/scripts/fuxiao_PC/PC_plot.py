import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

file_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/conceptgraphs_short/exps/r_mapping_stride2_short/pcd_r_mapping_stride2_short.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("Type of 'objects':", type(data['objects']))
print("First item in 'objects':", data['objects'][0])  # Check the structure of the first item



# Assuming `data` has been loaded as per previous instructions

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract 3D coordinates of objects
xs, ys, zs = zip(*data['objects'])  # Assuming 'objects' is a list of [x, y, z] coordinates

# Map class colors to object points
class_colors = data['class_colors']
object_colors = [class_colors[str(i % len(class_colors))] for i in range(len(xs))]  # Repeats if there are more objects than colors

# Plot the points with colors
for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
    color = object_colors[i]  # RGB tuple for each object
    ax.scatter(x, y, z, color=color, label=data['class_names'][i % len(data['class_names'])])

# Add edges
for edge in data['edges']:
    p1, p2 = edge
    x_vals = [xs[p1], xs[p2]]
    y_vals = [ys[p1], ys[p2]]
    z_vals = [zs[p1], zs[p2]]
    ax.plot(x_vals, y_vals, z_vals, 'gray')

# Label and show plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
