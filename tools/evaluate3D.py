import argparse
from sklearn.metrics import adjusted_rand_score as adjusted_rand_index
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from tifffile import imread as tifread


parser = argparse.ArgumentParser()
parser.add_argument("--predicted", required=True, help="path/files for predicted labels")
parser.add_argument("--true", required=True, help="path/files for true labels")
parser.add_argument("--output", required=False, help="output path/files")
parser.add_argument("--threshold", type=int, default=127, help="threshold for the predicted label")

a = parser.parse_args()


def unravel(true_matrix, predicted_matrix):
    'unravel 3d tif'
    true = true_matrix.ravel()
    pred = predicted_matrix.ravel()

    return true, pred


def matrix(true, pred):
    'initiating dictionary container'
    overlap = Counter(zip(true, pred))
    'list values of dicitonary'
    data = list(overlap.values())
    'discard index with less than 10 pixels'
    keep = np.array(data) > 10
    'row and col index with pixels greater than 10'
    row_ind, col_ind = zip(*overlap.keys())
    row_ind = np.array(row_ind)[keep]
    col_ind = np.array(col_ind)[keep]
    data = np.array(data)[keep]
    'initiating csr matrix'
    p_ij = csr_matrix((data, (row_ind, col_ind)))

    'calculating split errors'
    split_pre = p_ij[1:, 1:].sign().sum(axis=1).ravel()
    split_pre = np.asarray(split_pre).flatten()
    split_pre2 = split_pre - 1
    split_pre2[split_pre2 < 0] = 0
    splits = split_pre2.sum()

    'calculating merge errors'
    merges_pre = p_ij[1:, 1:].sign().sum(axis=0).ravel()
    merges_pre = np.asarray(merges_pre).flatten()
    merges_pre2 = merges_pre - 1
    merges_pre2[merges_pre2 < 0] = 0
    merges = merges_pre2.sum()

    ari = adjusted_rand_index(true, pred)

    return splits, merges, ari


def recall_precision(true, pred):
    tp, fn, fp = 0, 0, 0

    'Binarizing both tif files'
    true = true >= 1
    true = true.astype(int)
    pred = pred >= 1
    pred = pred.astype(int)

    'counting true positives, false positives and false negatives'
    for i in range(0, len(true)):
        if true[i] + pred[i] == 2:
            tp = tp + 1
        elif true[i] != pred[i] and true[i] == 1:
            fn = fn + 1
        elif true[i] != pred[i] and true[i] == 0:
            fp = fp + 1

    'calculating recall and precision'
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    return recall, precision


def main():

    'reading tif files'
    true_label = tifread(a.true)
    pred_label = tifread(a.predicted)

    'unravel tif files'
    true, pred = unravel(true_label, pred_label)

    'calculation of ari, split and merge errors and all pa'
    splits, merges, ari = matrix(true, pred)
    recall, precision = recall_precision(true, pred)

    'prints'
    print("\nEvaluation results:\n")
    print("Splits                   = %i" %splits)
    print("Merges                   = %i" %merges)
    print("Adjusted Rand Index      = %0.5f" %ari)
    print("Recall                   = %0.5f" %recall)
    print("Precision                = %0.5f\n" %precision)


main()
