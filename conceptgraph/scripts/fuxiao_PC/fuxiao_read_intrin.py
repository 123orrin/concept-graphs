import numpy as np

# Path to your .npy file
file_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_3_4/intrinsics/000000.npy'

# Load the .npy file
data = np.load(file_path)

# Display the data
print("Data:\n", data)

# Optionally check the shape and data type
print("Shape:", data.shape)
print("Data Type:", data.dtype)
