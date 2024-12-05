import pickle

file_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/conceptgraphs_short/exps/mapping_ex1/pcd_mapping_ex1.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

#print(type(data))


    print(data.keys())
print ("Number of objects:", len(data['objects']))

# Check all objects for their properties
for obj in data['objects']:
    #print(obj.keys()) 
    
    print("Class Name:", {obj['class_name']})
    print("color:", obj['pcd_color_np'])

    #break  # Remove break to check more objects if needed


#print(type(data['objects']), len(data['objects']))  
#print(type(data['edges']), len(data['edges']))      
#print(data['class_names'])                          
#print(data['class_colors'])                         # Inspect 'class_colors'
