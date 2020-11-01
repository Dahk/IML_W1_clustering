# Clustering exercise

## How to run

Run clustering algorithms and print metrics:
```bash
python3 main.py run -d DATASET_NAME
```

Run PCA on a dataset:
```bash
python3 main.py pca -d DATASET_NAME -c N_COMPONENTS
```

Run NearestNeighbors and order by distance:
```bash
python3 main.py nn-knee -d DATASET_NAME -n N_NEIGHBORS
```


Run KMeans with different cluster sizes [1, k] on a dataset:
```bash
python3 main.py km-knee -d DATASET_NAME -k UP_TO_K
```



\
\
\
Datasets reference:\
DuBois, Christopher L. & Smyth, P. (2008). [UCI Network Data Repository](http://networkdata.ics.uci.edu). Irvine, CA: University of California, School of Information and Computer Sciences.