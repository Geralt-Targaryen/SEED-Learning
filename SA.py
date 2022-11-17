from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import svm
import scipy.io as sio
import numpy as np
import time
import os
import argparse
from tqdm import trange


FOLDS = 15
EEG_X = sio.loadmat('./SEED_III/EEG_X.mat')['X'][0]  # 15 x 3394 x 310
EEG_Y = sio.loadmat('./SEED_III/EEG_Y.mat')['Y'][0]  # 15 x 3394 x 1


def train(fold, n_dimension, C, kernel, ACC):

    print(f"Fold {fold}, PID: {os.getpid()}")
    print(f"Fold {fold}, PID: {os.getpid()}", file=log)
    tic = time.time()

    prediction_all = []  # prediction of all 14 classifiers within one fold

    for k in trange(FOLDS - 1):
        source = (fold + k + 1) % FOLDS
        x_s = EEG_X[source]     # 3394 x 310
        x_t = EEG_X[fold]       # 3394 x 310
        y_s = np.ravel(EEG_Y[source])
        y_t = EEG_Y[fold]

        pca_s = PCA(n_components=n_dimension)
        pca_t = PCA(n_components=n_dimension)
        pca_s.fit(x_s)
        pca_t.fit(x_t)

        p_s = pca_s.components_     # n_dimension x 310
        p_t = pca_t.components_     # n_dimension x 310

        x_s = p_t.dot(p_s.T).dot(p_s).dot(x_s.T).T      # 3394 x n_dimension
        x_t = p_t.dot(x_t.T).T                          # 3394 x n_dimension

        if kernel == 'linear':
            clf = svm.LinearSVC(C=C)
        else:
            clf = svm.SVC(kernel=kernel, C=C)

        clf.fit(x_s, y_s)
        prediction = clf.predict(x_t)                   # (3394, )

        prediction_all.append(prediction.reshape((-1, 1)))

    prediction_all = np.concatenate(prediction_all, axis=1)
    # 3394 x 14

    prediction = [np.argmax(np.bincount(prediction_all[i, :] + 1)) - 1 for i in range(prediction_all.shape[0])]
    # 3394 x 1
    acc = accuracy_score(y_t, prediction)
    print('test accuracy of fold %d: %f' % (fold, acc), file=log)

    toc = time.time()
    print(f'Training time of fold {fold}: %f' % (toc - tic), file=log)

    ACC.append(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', required=True, help='the regular constant', type=float)
    parser.add_argument('--kernel', required=True, help='the kernel function chosen', type=str)
    parser.add_argument('--n_dimension', default=64, help='the dimension of aligned subspace', type=int)
    args = parser.parse_args()

    tic = time.time()

    ACC = []
    log = open(f'./Results/logs/train_SA_{args.kernel}_{args.C}_log.txt', 'w')

    for fold in range(FOLDS):
        train(fold, args.n_dimension, args.C, args.kernel, ACC)

    toc = time.time()
    print(f'Training time: {toc - tic}', file=log)
    print(f'Average accuracy of all folds: {np.mean(ACC)}', file=log)
    log.close()

