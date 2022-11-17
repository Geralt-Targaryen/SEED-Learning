from sklearn.metrics import accuracy_score
from numpy.ma import cov
from numpy import linalg
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


def train(fold, C, kernel, ACC):

    print(f"Fold {fold}, PID: {os.getpid()}")
    print(f"Fold {fold}, PID: {os.getpid()}", file=log)
    tic = time.time()

    prediction_all = []  # prediction of all 14 classifiers within one fold

    for k in trange(FOLDS - 1):
        source = (fold + k + 1) % FOLDS
        x_s = EEG_X[source]     # 3394 x 310
        x_t = EEG_X[fold]       # 3394 x 310
        x_s -= x_s.mean(axis=0)
        x_t -= x_t.mean(axis=0)

        y_s = np.ravel(EEG_Y[source])
        y_t = EEG_Y[fold]

        c_s = cov(x_s.T)        # 310 x 310
        c_t = cov(x_t.T)        # 310 x 310
        r = min(linalg.matrix_rank(c_s), linalg.matrix_rank(c_t))

        u_s, sigma_s, _ = linalg.svd(c_s, full_matrices=False)      # columns are singular vectors?
        u_t, sigma_t, _ = linalg.svd(c_t, full_matrices=False)

        A = u_s.dot(np.diag(sigma_s ** -0.5)).dot(u_s.T).dot(u_t[:, :r]).dot(np.diag(sigma_t[:r] ** 0.5)).dot(u_t[:, :r].T)

        x_s = A.T.dot(x_s.T).T

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
    args = parser.parse_args()

    log = open(f'./Results/logs/train_CORAL_{args.kernel}_{args.C}_log.txt', 'w')

    tic = time.time()
    ACC = []

    for fold in range(FOLDS):
        train(fold, args.C, args.kernel, ACC)

    toc = time.time()
    print(f'Training time: {toc - tic}', file=log)
    print(f'Average accuracy of all folds: {np.mean(ACC)}', file=log)

    log.close()



