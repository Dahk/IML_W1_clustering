import itertools
import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt


def best_accuracy(y_true, y_pred, max_iter=1e4):
    score1, _ = best_permutation(y_true, y_pred, max_iter=max_iter)
    score2 = max_accuracy(y_true, y_pred)

    return score1 if score1 > score2 else score2


def best_labelling(y_true, y_pred, max_iter=1e4):
    """
    Return the best labelling in terms of accuracy as a result
    of attempting max_iter permutations (brute force) and a search
    that matches labels in order from best correspondance
    """
    score1, perm1 = best_permutation(y_true, y_pred, max_iter=max_iter)
    #score2, perm2 = max_accuracy(y_true, y_pred)

    perm2 = max_accuracy2(y_true, y_pred)
    score2 = metrics.accuracy_score(y_true, y_pred)

    return perm1 if score1 > score2 else perm2

def best_permutation(y_true, y_pred, score_func=metrics.accuracy_score, greatest=True, 
                     use_pred_labels=False, max_iter=1e4):
    """
    Brute force compute the best score from all possible
    permutations of labels of y_pred

    Parameters
    ----------
    score_func : function to score
        Will be called as score_func(y_true, y_pred)
    greatest : bool
        Find greatest score otherwise lowest
    use_pred_labels : bool
        Whether to use the original labels on y_pred or rearrange new ones
        from 0 to n = len(unique_labels)
    max_iter : int
        Limit the maximum amount of permutations tried

    """
    pred_labels = np.unique(y_pred)

    if greatest:
        is_better = np.greater
    else:
        is_better = np.less

    if use_pred_labels:
        perm_labels = pred_labels
    else:
        perm_labels = np.arange(len(pred_labels))

    # factorial of 9 is too high to try every possibility in order
    if len(perm_labels) > 8:    
        # this generates random permutations
        def permutate(x):
            yield np.random.permutation(x)
    else:
        # this generates all possible permutations
        permutate = itertools.permutations

    y_perm = np.zeros_like(y_pred)
    best_score, best_perm = (0, 0)
    curr_iter = 0

    for perm in permutate(perm_labels):
        for pred_label, perm_label in zip(pred_labels, perm):
            y_perm[y_pred == pred_label] = perm_label
        
        score = score_func(y_true, y_perm)
        if is_better(score, best_score):
            best_score, best_perm = score, y_perm.copy()

        curr_iter += 1
        if curr_iter >= max_iter:
            break
    
    return best_score, best_perm


def max_accuracy2(y_true, y_pred):
    """
    Relabel the predicted labels *in order* to 
    achieve the best labelling
    """
    pairs = []
    unmatched_pred_labels = np.unique(y_pred).tolist()
    unmatched_true_labels = np.unique(y_true).tolist()

    for _ in range(len(unmatched_pred_labels)):
        best_match_count = -1
        best_match = (0, 0)
        for pred_label in unmatched_pred_labels:
            for true_label in unmatched_true_labels:
                test_cluster = np.full_like(y_pred, -1)
                test_cluster[y_pred == pred_label] = true_label

                match_count = np.count_nonzero(test_cluster == y_true)
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_match = (pred_label, true_label)
        
        unmatched_pred_labels.remove(best_match[0])
        unmatched_true_labels.remove(best_match[1])
        pairs.append(best_match)

    best_labelling = np.zeros_like(y_pred)
    for label, new_label in pairs:
        best_labelling[y_pred == label] = new_label
    
    return best_labelling


def max_accuracy(c1, c2):
    #c1: numpy array with label of cluster
    #c2: numpy array with label of true cluster
    
    c1 = c1.astype(str)
    c2 = c2.astype(str)
    match_satimage = pd.DataFrame({"Guess": c1, "True": c2})

    match_satimage['match'] = match_satimage['Guess'] + '-t' + match_satimage['True']
    comparison = pd.DataFrame(match_satimage['match'])

    A = comparison.value_counts()

    sum = 0
    clusters = []
    c1new = np.copy(c1).astype(int)
    j = 0
    for i in range(len(A)):
        C_str = A[[i]].index.values[0][0]
        #print(C_str)
        CTL = C_str.split('-')
        if CTL[0] in clusters or CTL[1] in clusters:
            pass
        else:
            c1new[c1 == CTL[0]] = CTL[1][1:]
            clusters.append(CTL[0])
            clusters.append(CTL[1])
            sum = sum + int(A[[i]])
            #print(clusters)
            #print(sum)
            j = j + 1

    accuracy = sum/len(c1)

    return accuracy, c1new.astype(int)


def print_binary_metrics(results):
    header = '{}\t{}\t\t{}\n'.format('accuracy', 'f-score', 'ARI')
    print(header)

    for name, y_true, y_pred in results:
        y_pred = best_labelling(y_true, y_pred)
        scores = []

        # accuracy
        scores.append(metrics.accuracy_score(y_true, y_pred))

        # f1_score
        scores.append(metrics.f1_score(y_true, y_pred))

        # ARI
        scores.append(metrics.cluster.adjusted_rand_score(y_true, y_pred))

        scores = [str(round(score, 2)) for score in scores]
        row = '\t\t'.join(scores + [name])
        print(row)


def print_multi_metrics(y_true, y_pred):
    pass


def plot_clusters(results, figsize=(9, 8)):
    results.append(('Ground truth', results[0][1], results[0][1]))

    fig, ax = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Cluster size')

    ax = ax.flat
    for i, (name, y_true, y_pred) in enumerate(results):
        x, y = np.unique(y_pred, return_counts=True)
        
        ax[i].bar(x, y, label='count')
        ax[i].title.set_text(name)
        ax[i].set_xlabel('Clusters')
        ax[i].set_ylabel('Number of instances')

        ax[i].yaxis.grid(color='0.85')
        ax[i].set_axisbelow(True)

    plt.subplots_adjust(wspace=0.4, hspace=0.35)
    plt.show()