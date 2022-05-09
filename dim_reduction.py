import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from nxcurve import quality_curve

KERNELS = ["linear", "rbf", "poly"]
FLOWER_FEATURES = "dev_flower_feature_vectors.npy"
FLOWER_CLASSES = "dev_flower_classes.npy"
FLOWER_CLASS_NAMES = ["daisy","dandelion", "rose", "sunflower", "tulip"]

ALZHEIMER_FEATURES = "dev_alzheimer_feature_vectors.npy"
ALZHEIMER_CLASSES = "dev_alzheimer_classes.npy"
ALZHEIMER_CLASS_NAMES = [ "Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"  ]
MARK_SIZE = 7

FEATURES = ALZHEIMER_FEATURES
CLASSES = ALZHEIMER_CLASSES
CLASS_NAMES = ALZHEIMER_CLASS_NAMES

#todo comparation based on separability (via kNN)
#todo comparation on crowding problem - distance
#todo ISOMAP

def pca_view(features, labels):
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(normalize(features))
    plt.title(f"PCA")
    plt.scatter(transformed[:,0], transformed[:,1], c = labels, s = MARK_SIZE)
    plt.show()
    eval_reduction(features, transformed, labels)

def kernel_pca_view(features, labels):
    for kernel in KERNELS:
        pca = KernelPCA(n_components=2)
        transformed = pca.fit_transform(normalize(features))
        plt.title(f"PCA with {kernel} kernel")
        plt.scatter(transformed[:,0], transformed[:,1], c = labels, s = MARK_SIZE)
        plt.show()


def t_sne_view(features, labels):
    #todo modify perplexity, default 30
    t_sne = TSNE(n_components=2, init = "pca")
    transformed = t_sne.fit_transform(normalize(features))
    plt.title(f"t-SNE")
    plt.scatter(transformed[:,0], transformed[:,1], c = labels, s = MARK_SIZE)
    plt.show()
    eval_reduction(features, transformed, labels)

def umap_view(features, labels):
    #todo modify perplexity, default 30
    um = umap.UMAP(n_components=2)
    transformed = um.fit_transform(normalize(features))
    plt.title(f"Umap")
    plt.scatter(transformed[:,0], transformed[:,1], c = labels, s = MARK_SIZE)
    plt.show()
    eval_reduction(features, transformed, labels)

def lda_view(features, labels):
    #uses classes - kind of cheating? 
    lda = LinearDiscriminantAnalysis(n_components=2)
    transformed = lda.fit_transform(normalize(features), labels)
    plt.title(f"lda (svd)")
    plt.scatter(transformed[:,0], transformed[:,1], c = labels, s = MARK_SIZE)
    plt.show()
    eval_reduction(features, transformed, labels)


def variance_threshold_view(features, labels):
    vt = VarianceThreshold()
    transformed = vt.fit_transform(features, labels)
    plt.title(f"variance thresholding")
    plt.scatter(transformed[:,0], transformed[:,1], c = labels, s = MARK_SIZE)
    plt.show()


def eval_reduction(data, data2D, labels):
    kNN = KNeighborsClassifier(n_neighbors = 5)
    kNN.fit(data2D, labels)
    predictions = kNN.predict(data2D)
    err_cout = np.sum(predictions != labels)
    print(f"k-NN Error rate of the method = {err_cout/len(labels)}")
    view_embedding_quality_curves(data, data2D)


def view_embedding_quality_curves(data, embedding, n_neighbors = 20):
    quality_curve(data,embedding, n_neighbors,'r',True)
    quality_curve(data,embedding, n_neighbors,'q',True)
    quality_curve(data,embedding, n_neighbors,'b',True)

feature_vectors = np.load(file = FEATURES)
classes = np.load(CLASSES)
pca_view(feature_vectors, classes)
t_sne_view(feature_vectors, classes)
kernel_pca_view(feature_vectors, classes)
umap_view(feature_vectors, classes)
lda_view(feature_vectors, classes)
#variance_threshold_view(feature_vectors, classes)