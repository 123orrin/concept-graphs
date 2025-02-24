"""
Assuming we have images files
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
import pickle as pkl
import gzip
import os

PRIMARY_PATH = '/home/hornylemur/repos/concept-graphs/conceptgraph/dataset/external'
SECONDARY_PATH = 'exps/s_detections_stride2_short/detections'

def load_features_by_class(folder, class_name):
    embeddings = None
    path = os.path.join(PRIMARY_PATH, folder, SECONDARY_PATH)
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dir in dirs:
        embedding = np.load(os.path.join(dir, 'image_feats.npz'))['arr_0']
        with gzip.open(os.path.join(dir, 'detection_class_labels.pkl.gz'), 'rb') as f:
            tmp = pkl.load(f)
            classes = []
            for cls in tmp:
                cls = ''.join([i for i in cls if not i.isdigit()])
                classes.append(cls[:-1])
            
            if class_name in classes:
                index = classes.index(class_name)
                if embeddings is None:
                    embeddings = embedding[index, :]
                else:
                    embeddings = np.vstack((embeddings, embedding[index, :]))
                classes.pop(index)
    return embeddings

def reduce_embeddings(embeddings, dim=3):
    pca = PCA(n_components=dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def plot_all(folders, class_name, dim=3):
    assert dim in [2, 3], 'dim must be 2 or 3'
    projection = '3d' if dim == 3 else None
    fig = plt.figure()
    ax = fig.add_subplot(projection=projection)
    legend = []

    for folder in folders:
        embeddings = load_features_by_class(folder, class_name)
        if embeddings is not None:
            n = embeddings.shape[0]
            reduced_embeddings = reduce_embeddings(embeddings, dim)
            if dim == 2:
                ax.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker='o', linestyle='')
            elif dim == 3:
                ax.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], marker='o', linestyle='')
        else:
            n = 0
            if dim == 2:
                ax.plot([], [], marker='o', linestyle='')
            elif dim == 3:
                ax.plot([], [], [], marker='o', linestyle='')
        legend.append(folder + f', n={n}')

    ax.legend(legend)
    plt.show()


DIM = 2
CLASS_NAME = 'chair' # ['chair', 'folded chair']
FOLDERS = ['embedding_chair_front', 'embedding_chair_back', 'embedding_chair_soccer_ball', 'embedding_chair_tennis_ball']

plot_all(FOLDERS, CLASS_NAME, DIM)