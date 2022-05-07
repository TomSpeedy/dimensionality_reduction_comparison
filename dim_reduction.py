import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

FLOWER_FEATURES = "dev_flower_feature_vectors.npy"
FLOWER_CLASSES = "dev_flower_classes.npy"
def pca_view(features, labels):
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(normalize(features))
    plt.scatter(transformed[:,0], transformed[:,1], c = labels)
    plt.show()

def t_sne_view(features, labels):
    t_sne = TSNE(n_components=2)
    transformed = t_sne.fit_transform(normalize(features))
    plt.scatter(transformed[:,0], transformed[:,1], c = labels)
    plt.show()

feature_vectors = np.load(file = FLOWER_FEATURES)
classes = np.load(FLOWER_CLASSES)
print(classes.ravel())
pca_view(features=feature_vectors, labels = classes)
t_sne_view(features = feature_vectors, labels = classes)