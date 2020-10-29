import sys
import click
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.metrics.cluster import contingency_matrix

from matplotlib import pyplot as plt

import datasets
from cluster import KMeans
from cluster import KMeanspp
from cluster import BisectingKMeans
from cluster import FuzzyCMeans
from evaluation import (print_binary_metrics,
                        print_multi_metrics)


@click.group()
def cli():
    pass


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
    kmeans = KMeans(k=2)
    y_pred = kmeans.fit_predict(X)
    results.append(('KMeans', y, y_pred))

    # BisectingKMeans
    bikmeans = BisectingKMeans(k=2)
    y_pred = bikmeans.fit_predict(X)
    results.append(('BiKMeans', y, y_pred))

    # KMeanspp
    kmeanspp = KMeanspp(k=2)
    y_pred = kmeanspp.fit_predict(X)
    results.append(('KMeanspp', y, y_pred))

    # FuzzyCMeans
    fuzzycmeans = FuzzyCMeans(c=2, m=2, random_state=35)
    y_pred = fuzzycmeans.fit_predict(X)
    results.append(('FuzzyCMeans', y, y_pred))

    # DBSCAN
    dbscan = DBSCAN(eps=1.751004016064257, min_samples=216)
    y_pred = dbscan.fit_predict(X)
    results.append(('DBSCAN', y, y_pred))

    print_binary_metrics(results)


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


@cli.command('knee')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
@click.option('-k', default=2, help='Maximum value of k to try [0, k]')
def run_knee(d, k):
    if d == 'kropt':
        run_knee_kropt(k)

    elif d == 'satimage':
        run_knee_satimage(k)

    elif d == 'credita':
        run_knee_credita(k)

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def run_knee_kropt(k):
    X, y = datasets.load_kropt()


def run_knee_satimage(k):
    X, y = datasets.load_satimage()


def run_knee_credita(k):
    X, y = datasets.load_credita()








if __name__ == "__main__":
    cli()

