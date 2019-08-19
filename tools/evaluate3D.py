import argparse
from sklearn.metrics import adjusted_rand_score as adjusted_rand_index
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
import glob
from tifffile import imread as tifread
from skimage.io import imread
import os

parser = argparse.ArgumentParser()
parser.add_argument("--predicted", required=True, help="path/files for predicted labels")
parser.add_argument("--true", required=True, help="path/files for true labels")
parser.add_argument("--output", required=False, help="output path/files")
parser.add_argument("--threshold", type=int, default=127, help="threshold for the predicted label")
parser.add_argument("--format", required=True, choices=["png", "tif"])

a = parser.parse_args()


def unravel(true_matrix, predicted_matrix):
    true = true_matrix.ravel()
    pred = predicted_matrix.ravel()

    return true, pred


def matrix(true, pred):
    ##########
    n = true.size
    ##########
    overlap = Counter(zip(true, pred))
    data = list(overlap.values())
    keep = np.array(data) > 10
    row_ind, col_ind = zip(*overlap.keys())
    row_ind = np.array(row_ind)[keep]
    col_ind = np.array(col_ind)[keep]
    data = np.array(data)[keep]
    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind)
    data = np.array(data)
    p_ij = csr_matrix((data, (row_ind, col_ind)))

    #print(p_ij)
    #print(p_ij.sum(axis=0))
    #print(p_ij.sum(axis=1))

    a_i = np.array(p_ij[1:, :].sum(axis=1))
    b_j = np.array(p_ij[1:, 1:].sum(axis=0))

    p_i0 = p_ij[1:, 0]
    #print(p_ij)
    p_ij = p_ij[1:, 1:]

    sumA = (a_i * a_i).sum()
    sumB = (b_j * b_j).sum() + p_i0.sum() / n
    sumAB = p_ij.multiply(p_ij).sum() + p_i0.sum() / n

    ISBI_rand = 1 - (sumA + sumB - 2 * sumAB) / (n ** 2)


    # if max(row_ind) > max(col_ind):
    #     counter_col = Counter(col_ind)
    #     col = [i for i in counter_col.values() if i > 1]
    #     splits = (np.array(col) - 1).sum()
    # else:
        #counter_row = Counter(row_ind)
        #row = [i for i in counter_row.values() if i > 1]
        #splits = (np.array(row) - 1).sum()

    # print(sorted(row_ind))
    # print(sorted(col_ind))
    # print(len(row_ind))
    # print(len(col_ind))
    #splits = (np.array(p_ij > 0).sum(axis=0)).sum() - np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum() - 1 - np.array(p_ij[1:, 1:].sum(axis=1) != p_ij[1:, 1:].max(axis=1)).sum()
    ############################## splits = np.array(p_ij[1:, 1:].sum(axis=1) != p_ij[1:, 1:].max(axis=1)).sum()

    # print(len(row_ind))
    # print(np.array(p_ij[1:, 1:].sum(axis=1)))
    # print(p_ij[1:, 1:].max(axis=1))
    # print(np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum())

    # counter_row = Counter(row_ind)
    # row = [i for i in counter_row.values() if i > 1]
    # splits = int((np.array(row) - 1).sum())

    '''
    splitsberechnung = 
    1.) np.array(p_ij[1:, 1:] > 0).sum(axis=0)).sum()  --> wieviele spalten (predicted) gibt es ... overlap + cytoplasm hat 408 und only_cytoplasm 400
    MINUS
    2.) np.array(p_ij[1:, 1:].sum(axis=1) != p_ij[1:, 1:].max(axis=1)).sum() --> wie oft min != max (alte splitfunktion) overlap + cytoplasm hat 6 und only_cytoplasm 0
    MINUS
    3.) np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum() --> gegenteil von splits overlap + cytoplasm hat 394 auf achse1 und only_cytoplasm hat 400
    
    DH ... 400 - 0 - 400 = 0 bei only_cytoplasm
    und .. 408 - 6 - 394 = 8 bei overlap + cytoplasm (8 ist der wahre Splitwert, weil es insgesamt 8 getrennte Pixelcluster gibt und ein Objekt 3x gesplittet wurde --> deswegen zeigt es 6 splits obwohl es 8 sind)
    '''

    #splits = np.array(p_ij[1:, 1:] > 0).sum(axis=1).sum() - np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum()

    #splits = (np.array(p_ij[1:, 1:] > 0).sum(axis=0)).sum() - np.array(p_ij[1:, 1:].sum(axis=1) != p_ij[1:, 1:].max(axis=1)).sum() - np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum()
    #splits = (np.array(p_ij[1:, 1:]) > 0).sum(axis=1).sum() - np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum()
    #print(p_ij[1:, 1:])
    #print(np.array(p_ij[1:, 1:].sum(axis=0)))
    #print(np.array(p_ij[1:, 1:].sum(axis=1)))
    #print(np.array(p_ij[1:, 1:].sum(axis=0)).sum())
    #print(np.array(p_ij[1:, 1:].sum(axis=0) == p_ij[1:, 1:].max(axis=0)).sum())

    #splits = (np.greater(np.array(p_ij[1:, 1:]), 0)).sum(axis=1).sum() - np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum()
    #splits = np.array(p_ij[1:, 1:].sum(axis=1) != p_ij[1:, 1:].max(axis=1)).sum()

    '''
    mergesberechnung = 
    1.) np.array(p_ij[1:, 1:] > 0).sum(axis=0)).sum() --> direkt vorne mit [1:, 1:] nullenzuweisungen bei objekten die hintergrund zugewiesen kriegen entfernen und sonst zählt es nur spalten
    MINUS
    2.) Anzahl von Objekten mit sum = max
    '''

    split_pre = p_ij[1:, 1:].sign().sum(axis=1).ravel()
    split_pre = np.asarray(split_pre).flatten()
    split_pre2 = split_pre - 1
    split_pre2[split_pre2 < 0] = 0
    splits = split_pre2.sum()

    merges_pre = p_ij[1:, 1:].sign().sum(axis=0).ravel()
    merges_pre = np.asarray(merges_pre).flatten()
    merges_pre2 = merges_pre - 1
    merges_pre2[merges_pre2 < 0] = 0
    merges = merges_pre2.sum()

    #perfect_object = 0
    #for (i, j) in zip(merges_pre, split_pre):
    #    if i == j:
    #        perfect_object = perfect_object + 1

    #splits = np.array(p_ij[1:, 1:].sum(axis=1) != p_ij[1:, 1:].max(axis=1)).sum()

    #merges = np.array(p_ij[1:, 1:] > 0).sum(axis=0).sum() - np.array(p_ij[1:, 1:].sum(axis=0) == p_ij[1:, 1:].max(axis=0)).sum()

    #merges = (np.greater(np.array(p_ij[1:, 1:]), 0)).sum(axis=0).sum() - np.array(p_ij[1:, 1:].sum(axis=0) == p_ij[1:, 1:].max(axis=0)).sum()

    #todo wv objekte korrekt vorhergesagt wurden

    #splits = np.array(p_ij[1:, 1:] > 0).sum(axis=0).sum() - np.array(p_ij[1:, 1:].sum(axis=1) == p_ij[1:, 1:].max(axis=1)).sum()

    #perfect_object = np.array(p_ij[1:, 1:].sum(axis=0) == p_ij[1:, 1:].max(axis=0)).sum()

    #merges_torsten = np.array(p_ij[1:, 1:].sum(axis=0) != p_ij[1:, 1:].max(axis=0)).sum()

    #print(np.where(np.array(p_ij[1:, 1:].sum(axis=0) == p_ij[1:, 1:].max(axis=0))))
    #print(np.where(np.array(p_ij[1:, 1:].sum(axis=0) != p_ij[1:, 1:].max(axis=0))))
    #print(np.where(np.array(p_ij[1:, 1:].sum(axis=1) != p_ij[1:, 1:].max(axis=1))))
    print(splits)
    #print(merges_alt)
    print(merges)

    ari = adjusted_rand_index(true, pred)

    return p_ij, splits, merges, ari, ISBI_rand


def recall_precision(true, pred):
    tp, fn, fp = 0, 0, 0

    true = true >= 1
    true = true.astype(int)
    pred = pred >= 1
    pred = pred.astype(int)

    for i in range(0, len(true)):
        if true[i] + pred[i] == 2:
            tp = tp + 1
        elif true[i] != pred[i] and true[i] == 1:
            fn = fn + 1
        elif true[i] != pred[i] and true[i] == 0:
            fp = fp + 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    return recall, precision


def main():
    '''matrices'''

    if a.format == "tif":
        pred_paths = sorted(glob.glob(a.predicted))
        true_paths = sorted(glob.glob(a.true))
    elif a.format == "png":
        pred_paths = sorted(glob.glob(os.path.join(a.predicted, "*.png")))
        true_paths = sorted(glob.glob(os.path.join(a.true, "*.png")))

    list_pred = []
    list_true = []

    for (pred_path, true_path) in zip(pred_paths, true_paths):
        if a.format == "tif":
            true_label = tifread(true_path)[:, :, :]
            pred_label = tifread(pred_path)[:, :, :]
        elif a.format == "png":
            try:
                true_label = imread(true_path)[:, :, 0]
                pred_label = imread(pred_path)[:, :, 0]
                list_true.append(true_label)
                list_pred.append(pred_label)
            except:
                true_label = imread(true_path)[:, :]
                pred_label = imread(pred_path)[:, :]
                list_true.append(true_label)
                list_pred.append(pred_label)

    #true = np.array([[1, 1, 2, 2], [0, 0, 0, 0], [0, 0, 3, 3]])
    #pred = np.array([[1, 1, 2, 0], [0, 0, 0, 0], [3, 3, 3, 4]])

    '''functions'''
    true, pred = unravel(true_label, pred_label)
    all_pairings, splits, merges, ari, ISBI_rand = matrix(true, pred)
    recall, precision = recall_precision(true, pred)

    '''prints'''
    print("Splits =", splits)
    print("Merges =", merges)
    print("Adjusted Rand Index =", ari)
    print("ISBI_RAND =", ISBI_rand)
    print("Recall =", recall)
    print("Precision =", precision)


main()
