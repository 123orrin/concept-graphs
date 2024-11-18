import pickle

file_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/conceptgraphs_short/exps/r_mapping_stride2_short/pcd_r_mapping_stride2_short.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(type(data))
# If it's a dictionary, list keys to understand the content
if isinstance(data, dict):
    print(data.keys())

print(type(data['objects']), len(data['objects']))  # Check the type and size of 'objects'
print(type(data['edges']), len(data['edges']))      # Check the type and size of 'edges'
print(data['class_names'])                          # Inspect 'class_names'
print(data['class_colors'])                         # Inspect 'class_colors'
