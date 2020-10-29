import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.cluster import KMeans as kmeans


def sse_l2(x, y):
    return np.sum((x - y) ** 2)


def _sanitize(X):
    return X.values if isinstance(X, pd.DataFrame) else X


class KMeans:

    def __init__(self, k, tol=1e-7, max_iter=100, n_init=10, random_state=None):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.centroids_ = None
    
        if random_state is not None:
            random_state = np.int64(random_state) 
        self.random_state = RandomState(seed=random_state)

    def fit(self, X):
        """
        Run the algorithm n_init times and keep
        the one with best inertia
        """
        if self.centroids_ is not None:
            centroids_init = self.centroids_.copy()
        else:
            centroids_init = None

        best = []
        for _ in range(self.n_init):
            self.centroids_ = centroids_init
            self._fit(X)

            if best == [] or self.inertia_ < best[0]:
                # save best results
                best = (
                    self.inertia_,
                    self.n_iter_,
                    self.labels_.copy(),
                    self.centroids_.copy(),
                    )

        # restore best results
        (self.inertia_,
         self.n_iter_,
         self.labels_,
         self.centroids_) = best
        return self


    def _fit(self, X):
        """
        Run main k-means algorithm to classify the data into
        k clusters
        """
        X = _sanitize(X)

        if self.centroids_ is None:
            # select k initial centroids randomly
            centroids_i = set()
            
            while len(centroids_i) != self.k:
                # ensure we generate a set of k unique centroids
                centroids_i = {self.random_state.randint(X.shape[0]) for _ in range(self.k)}

            centroids = X[list(centroids_i)]
        else:
            centroids = self.centroids_

        curr_iter = 0
        variance_diff = self.tol * 2
        variances = np.zeros(self.k)

        while curr_iter < self.max_iter and np.any(variance_diff > self.tol):
            # assign each point the closest centroid
            labels = self._assign(X, centroids)
            self._fix_empty(X, labels, centroids)

            # calculate new centroids
            for i in range(self.k):
                cluster = X[labels == i]
                new_centroid = np.mean(cluster, axis=0)
                centroids[i] = new_centroid

            # calculate difference in variance from previous clustering
            variances_old = variances
            variances = np.empty_like(variances_old)
            for i in range(self.k):
                cluster = X[labels == i]
                var = np.var(cluster, axis=0)
                variances[i] = np.linalg.norm(var, 2)

            variance_diff = variances - variances_old

            curr_iter += 1

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = self._inertia(X, labels, centroids)
        self.n_iter_ = curr_iter
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, X):
        if self.centroids_ is None:
            raise ValueError('This model is not fitted yet.')
        
        X = _sanitize(X)
        labels = self._assign(X, self.centroids_)
        return labels

    def _assign(self, X, centroids):
        """
        Assign each sample to the cluster of the closest centroid
        """
        # calculate distances to centroids
        centroid_dists = np.ndarray((X.shape[0], self.k))
        for i in range(self.k):
            # here einsum = sum squared factors
            diff = X - centroids[i]
            centroid_dists[:, i] = np.sqrt(np.einsum('ij,ij->i', diff, diff))   

        return np.argmin(centroid_dists, axis=1)

    def _fix_empty(self, X, labels, centroids):
        """
        Fill/fix empty clusters if there are any by taking a random
        sample from the cluster with highest SSE and using it as a new
        centroid for the empty cluster, while ensuring that centroid 
        was not already in use.
        """
        counts = np.array([np.count_nonzero(labels == i) for i in range(self.k)])
        
        # while there is an empty cluster
        while np.any(counts == 0):
            # index of empty cluster
            empty_i = np.argmin(counts)

            # compute SSE
            cluster_sses = []
            for i in range(self.k):
                if i != empty_i:
                    cluster = X[labels == i]
                    sse = sse_l2(cluster, cluster.mean())
                    cluster_sses.append(sse)
                else:
                    cluster_sses.append(0)

            # choose as new centroid a random point
            # from the cluster with highest SSE
            highest_i = np.argmax(cluster_sses)

            random_i = self.random_state.randint(counts[highest_i])
            new_centroid = X[labels == highest_i][random_i]

            # if it is already used as centroid, look for another one
            already_used = np.any([np.array_equal(new_centroid, c) for c in centroids])
            while already_used:
                random_i = self.random_state.randint(counts[highest_i])
                new_centroid = X[labels == highest_i][random_i]
                already_used = np.any([np.array_equal(new_centroid, c) for c in centroids])

            # reassign centroid
            labels[np.where(labels == highest_i)[0][random_i]] = empty_i
            centroids[empty_i] = new_centroid

            counts = np.array([np.count_nonzero(labels == i) for i in range(self.k)])

    def _inertia(self, X, labels, centroids):
        """
        Calculate the inertia, which is the global SSE
        of distances from each sample to its centroid
        """
        inertia = 0
        for i in range(self.k):
            cluster = X[labels == i]
            centroid = centroids[i]
            inertia += sse_l2(cluster, centroid)

        return inertia


class BisectingKMeans:

    def __init__(self, k, tol=1e-7, max_iter=100, n_init=10, random_state=None):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X):
        X = _sanitize(X)

        labels = np.zeros(X.shape[0], dtype=np.int)
        clusters = [(sse_l2(X, X.mean()), 0)]

        for i in range(1, self.k):
            target_label = self._pick(clusters)
            cluster_mask = labels == target_label

            bicluster = KMeans(2, self.tol, self.max_iter, self.n_init,
                self.random_state).fit_predict(X[cluster_mask])

            # mark one half of the new partition with a new label
            # we can keep the old label for the other half
            new_cluster_mask = np.where(cluster_mask)[0][bicluster == 0]
            labels[new_cluster_mask] = i

            self._insert(target_label, i, clusters, labels, X)

        self.labels_ = labels
        self.inertia_ = sum(c[0] for c in clusters)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def _insert(self, label_a, label_b, clusters, labels, X):
        # calculate SSE
        cluster_a = X[labels == label_a]
        cluster_b = X[labels == label_b]
        sse_0 = sse_l2(cluster_a, cluster_a.mean())
        sse_1 = sse_l2(cluster_b, cluster_b.mean())

        # add to list
        clusters.append((sse_0, label_a))
        clusters.append((sse_1, label_b))

    def _pick(self, clusters):
        # pick and remove the cluster with less cohesion (highest SSE)
        i = np.argmax([c[0] for c in clusters])
        return clusters.pop(i)[1]


class KMeanspp(KMeans):

    def _fit(self, X):
        if self.centroids_ is None:
            # run improved init only when not fitted already
            X = _sanitize(X)

            # select first random centroid
            centroids = np.ndarray((self.k, X.shape[1]))
            rand_i = self.random_state.randint(X.shape[0])
            centroids[0] = X[rand_i]

            dists = np.ndarray((X.shape[0], self.k))
            for i in range(1, self.k):
                # compute squared distances to all centroids
                for j in range(i):
                    diff = X - centroids[j]
                    dists[:, j] = np.einsum('ij,ij->i', diff, diff)

                # pick as centroid the point with largest minimum distance
                # to any centroid
                min_dists = np.min(dists[:, :i], axis=1)
                farthest_point_i = np.argmax(min_dists)
                centroids[i] = X[farthest_point_i]

            self.centroids_ = centroids

        return super()._fit(X)


class FuzzyCMeans:

    def __init__(self, c, m=2, eps=1e-7, max_iter=100, n_init=10, random_state=None):
        self.c = c
        self.m = m
        self.eps = eps
        self.max_iter = max_iter
        self.n_init = n_init
        self.V_ = None

        if random_state is not None:
            random_state = np.int64(random_state) 
        self.random_state = RandomState(seed=random_state)
    
    def fit(self, X):
        """
        Run the algorithm n_iter times and keep
        the one with best inertia
        """
        if self.V_ is not None:
            V_init = self.V_.copy()
        else:
            V_init = None

        best = []
        for _ in range(self.n_init):
            self.V_ = V_init
            self._fit(X)

            if best == [] or self.perfindex_ < best[0]:
                # save best results
                best = (
                    self.perfindex_,
                    self.n_iter_,
                    self.labels_.copy(),
                    self.U_.copy(),
                    self.V_.copy(),
                    )

        # restore best results
        (self.perfindex_,
         self.n_iter_,
         self.labels_,
         self.U_,
         self.V_) = best
        return self

    def _fit(self, X):
        """
        Run main fuzzy c-means algorithm to classify the data into
        c labels or clusters
        """
        X = _sanitize(X)

        if self.V_ is None:
            # select c initial centroids randomly
            centroids_i = set()
            while len(centroids_i) != self.c:
                # ensure we generate a set of c unique centroids
                centroids_i = {self.random_state.randint(X.shape[0]) for _ in range(self.c)}

            V = X[list(centroids_i)] * 1.001    # avoid divisions by 0
        else:
            V = self.V_

        curr_iter = 0
        V_diff = self.eps * 2

        while curr_iter < self.max_iter and np.any(V_diff > self.eps):
            # compute U
            U = self._compute_U(X, V)
            
            # compute V
            V_old = V
            V = self._compute_V(X, U)

            V_diff = np.linalg.norm(V - V_old, 2, axis=1)
            curr_iter += 1

        self.U_ = U
        self.V_ = V
        self.labels_ = np.argmax(U, axis=0)
        self.perfindex_ = self._performance_index(X, U, V)
        self.n_iter_ = curr_iter
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, X):
        if self.V_ is None:
            raise ValueError('This model is not fitted yet.')

        X = _sanitize(X)
        
        V = self.V_
        U = self._compute_U(X, V)
        return np.argmax(U, axis=0)        

    def _compute_U(self, X, V):
        # precompute distances to centroids
        dists = np.empty((self.c, X.shape[0]))
        for i in range(self.c):
            diff = X - V[i]
            dists[i, :] = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        
        # init with 1s, it will always be at least 1
        # since we always divide the distance to one cluster
        # by itelf for every cluster
        U = np.ones_like(dists)
        exp = 2 / (self.m - 1)

        # main formula broadcasted
        for i in range(self.c):
            for j in range(self.c):
                if i != j:  # thats why we init with 1s
                    U[i] += (dists[i] / dists[j]) ** exp

        return U ** -1.0

    def _compute_V(self, X, U):
        V = np.empty((self.c, X.shape[1]))
        Um = U ** self.m

        # main formula broadcasted
        for i in range(self.c):
            Um_col = Um[i].reshape(-1, 1)
            V[i] = np.sum(Um_col * X, axis=0) / np.sum(Um_col)
            
        return V

    def _performance_index(self, X, U, V):
        # precompute errors
        inner_errors = np.empty((self.c, X.shape[0]))
        for i in range(self.c):
            # here einsum = sum of squared factors
            diff = X - V[i]
            inner_errors[i, :] = np.einsum('ij,ij->i', diff, diff)

        global_mean = X.mean(axis=0)
        diff = V - global_mean
        outer_errors = np.einsum('ij,ij->i', diff, diff).reshape(-1, 1)     

        # main formula
        perfindex = np.sum(U * (inner_errors - outer_errors))
        return perfindex

    
