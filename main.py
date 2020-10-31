import sys
import click
import numpy as np
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import datasets
from cluster import KMeans
from cluster import KMeanspp
from cluster import BisectingKMeans
from cluster import FuzzyCMeans
from evaluation import (print_binary_metrics,
                        print_multi_metrics,
                        plot_clusters)


RS_CREDITA_KMEANS = 21320
RS_CREDITA_BIKMEANS = 21320
RS_CREDITA_KMEANSPP = 90184
RS_CREDITA_FUZZYCMEANS = 80723


@click.group()
def cli():
    pass


# -----------------------------------------------------------------------------------------
# Run all agorithms and plot/print metrics
@cli.command('run')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
def run(d):
    if d == 'kropt':
        run_kropt()

    elif d == 'satimage':
        run_satimage()

    elif d == 'credita':
        run_credita()

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def run_kropt():
    X, y = datasets.load_kropt()


def run_satimage():
    X, y = datasets.load_satimage()


def run_credita():
    X, y = datasets.load_credita()
    results = []

    # KMeans
    kmeans = KMeans(k=2, random_state=RS_CREDITA_KMEANS)
    y_pred = kmeans.fit_predict(X)
    results.append(('KMeans', y, y_pred))

    # BisectingKMeans
    bikmeans = BisectingKMeans(k=2, random_state=RS_CREDITA_BIKMEANS)
    y_pred = bikmeans.fit_predict(X)
    results.append(('BiKMeans', y, y_pred))

    # KMeanspp
    kmeanspp = KMeanspp(k=2, random_state=RS_CREDITA_KMEANSPP)
    y_pred = kmeanspp.fit_predict(X)
    results.append(('KMeanspp', y, y_pred))

    # FuzzyCMeans
    fuzzycmeans = FuzzyCMeans(c=2, m=2, random_state=RS_CREDITA_FUZZYCMEANS)
    y_pred = fuzzycmeans.fit_predict(X)
    results.append(('FuzzyCMeans', y, y_pred))

    # DBSCAN
    #dbscan = DBSCAN(eps=1.751004016064257, min_samples=216)
    dbscan = DBSCAN(eps=1.75, min_samples=10)
    y_pred = dbscan.fit_predict(X)
    y_pred[y_pred == -1] = 1
    results.append(('DBSCAN', y, y_pred))
    
    print_binary_metrics(results)
    plot_clusters(results)


# -----------------------------------------------------------------------------------------
# PCA
@cli.command('pca')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
@click.option('-c', default=2, help='# components to reduce to')
def run_pca(d, c):
    if d == 'kropt':
        run_pca_kropt(c)

    elif d == 'satimage':
        run_pca_satimage(c)

    elif d == 'credita':
        run_pca_credita(c)

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def run_pca_kropt(c):
    X, y = datasets.load_kropt()


def run_pca_satimage(c):
    X, y = datasets.load_satimage()


def run_pca_credita(c):
    X, y = datasets.load_credita()
    pca = PCA(n_components=c)
    pca.fit(X)
    X = pca.transform(X)

    for i in np.unique(y):
        plt.scatter(*[X[y == i, j] for j in range(c)], color='C'+str(i))
    plt.show()

# -----------------------------------------------------------------------------------------
# DBSCAN - Nearest Neighbours knee  
@cli.command('nn-knee')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
@click.option('-n', default=10, help='n nearest neighbors')
def run_nn_knee(d, n):
    if d == 'kropt':
        run_nn_knee_kropt(n)

    elif d == 'satimage':
        run_nn_knee_satimage(n)

    elif d == 'credita':
        run_nn_knee_credita(n)

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def run_nn_knee_kropt(n):
    X, y = datasets.load_kropt()

    nbrs = NearestNeighbors(n_neighbors=n).fit(X)
    distances, indices = nbrs.kneighbors(X)
    sorted_dists = np.sort(distances[:, -1])

    plt.plot(range(X.shape[0]), sorted_dists)
    plt.title(f'Distance to n={n} nearest neighbor')
    plt.ylabel('Distance')
    plt.xlabel('Ordered instances')
    plt.axes().yaxis.grid(color='0.85')
    plt.axes().set_axisbelow(True)
    plt.show()


def run_nn_knee_satimage(n):
    X, y = datasets.load_satimage()

    nbrs = NearestNeighbors(n_neighbors=n).fit(X)
    distances, indices = nbrs.kneighbors(X)
    sorted_dists = np.sort(distances[:, -1])

    plt.plot(range(X.shape[0]), sorted_dists)
    plt.title(f'Distance to n={n} nearest neighbor')
    plt.ylabel('Distance')
    plt.xlabel('Ordered instances')
    plt.axes().yaxis.grid(color='0.85')
    plt.axes().set_axisbelow(True)
    plt.show()


def run_nn_knee_credita(n):
    X, y = datasets.load_credita()

    nbrs = NearestNeighbors(n_neighbors=n).fit(X)
    distances, indices = nbrs.kneighbors(X)
    sorted_dists = np.sort(distances[:, -1])

    plt.plot(range(X.shape[0]), sorted_dists)
    plt.title(f'Distance to n={n} nearest neighbor')
    plt.ylabel('Distance')
    plt.xlabel('Ordered instances')
    plt.axes().yaxis.grid(color='0.85')
    plt.axes().set_axisbelow(True)
    plt.show()
    

# -----------------------------------------------------------------------------------------
# KMeans - K clusters knee
@cli.command('km-knee')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
@click.option('-k', default=10, help='Maximum value of k to try [1, k]')
def run_km_knee(d, k):
    if d == 'kropt':
        run_km_knee_kropt(k)

    elif d == 'satimage':
        run_km_knee_satimage(k)

    elif d == 'credita':
        run_km_knee_credita(k)

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def run_km_knee_kropt(k):
    X, y = datasets.load_kropt()

    cohesion = []
    for _ in range(1, k):
        kmeans = KMeanspp(k=k, n_init=10).fit(X)
        cohesion.append(kmeans.cohesion_)
    
    plt.plot(range(k - 1), cohesion)
    plt.title('Cluster cohesion with respect to k\nKmeanspp / Kropt')
    plt.xlabel('k')
    plt.ylabel('Cohesion value')
    plt.axes().yaxis.grid(color='0.85')
    plt.axes().set_axisbelow(True)
    plt.show()


def run_km_knee_satimage(k):
    X, y = datasets.load_satimage()

    cohesion = []
    for _ in range(1, k):
        kmeans = KMeanspp(k=k, n_init=10).fit(X)
        cohesion.append(kmeans.cohesion_)
    
    plt.plot(range(k - 1), cohesion)
    plt.title('Cluster cohesion with respect to k\nKmeanspp / SatImage')
    plt.xlabel('k')
    plt.ylabel('Cohesion value')
    plt.axes().yaxis.grid(color='0.85')
    plt.axes().set_axisbelow(True)
    plt.show()


def run_km_knee_credita(k):
    X, y = datasets.load_credita()

    cohesion = []
    for _ in range(1, k):
        kmeans = KMeanspp(k=k, n_init=50).fit(X)
        cohesion.append(kmeans.cohesion_)
    
    plt.plot(range(k - 1), cohesion)
    plt.title('Cluster cohesion with respect to k\nKmeanspp / Credit-A')
    plt.xlabel('k')
    plt.ylabel('Cohesion value')
    plt.axes().yaxis.grid(color='0.85')
    plt.axes().set_axisbelow(True)
    plt.show()


if __name__ == "__main__":
    cli()

