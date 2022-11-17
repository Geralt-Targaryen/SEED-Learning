from sklearn.metrics import accuracy_score, pairwise
from cvxopt import matrix, solvers
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


def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


class KMM:
    # from: https://github.com/jindongwang/transferlearning/tree/master/code/traditional
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta


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

        kmm = KMM(kernel_type='linear', B=1)
        beta = kmm.fit(x_s, x_t)
        x_s_new = beta * x_s

        if kernel == 'linear':
            clf = svm.LinearSVC(C=C)
        else:
            clf = svm.SVC(kernel=kernel, C=C)

        clf.fit(x_s_new, y_s)
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

    log = open(f'./Results/logs/train_KMM_{args.kernel}_{args.C}_log.txt', 'w')

    tic = time.time()
    ACC = []

    for fold in range(FOLDS):
        train(fold, args.C, args.kernel, ACC)

    toc = time.time()
    print(f'Training time: {toc - tic}', file=log)
    print(f'Average accuracy of all folds: {np.mean(ACC)}', file=log)

    log.close()



