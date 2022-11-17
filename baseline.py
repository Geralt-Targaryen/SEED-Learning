from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle
import scipy.io as sio
import numpy as np
import time
import argparse
from tqdm import trange
from multiprocessing import Pool

FOLDS = 15


def train_single_svm(x_y_k_C):
    if x_y_k_C[2] == 'linear':
        clf = svm.LinearSVC(C=x_y_k_C[3])
    else:
        clf = svm.SVC(kernel=x_y_k_C[2], C=x_y_k_C[3])

    clf.fit(x_y_k_C[0], x_y_k_C[1])

    return clf


def train(dataset_X:np.ndarray, dataset_Y:np.ndarray, C:float, kernel:str, save:bool) -> list:
    train_log = open(f'./Results/logs/train_svm_{kernel}_{C}_log.txt', 'w')
    print(f"train {kernel} svm with C={C}")

    # Cross Validation
    tic = time.time()
    with Pool(FOLDS) as pool:
        clfs = list(pool.map(
            train_single_svm,
            [(np.concatenate([dataset_X[(k + i + 1) % FOLDS] for i in range(FOLDS-1)]),
              np.ravel(np.concatenate([dataset_Y[(k + i + 1) % FOLDS] for i in range(FOLDS-1)])),
              kernel,
              C
              ) for k in range(FOLDS)]
        ))

    toc = time.time()
    print('training time: %f'%(toc-tic), file=train_log)

    if save:
        for k in range(FOLDS):
            model_file = open(f'./Results/models/svm_{kernel}_{C}_{k}.pickle', 'wb')
            pickle.dump(clfs[k], model_file)
            model_file.close()

    train_log.close()
    return clfs


def test(dataset_X:np.ndarray, dataset_Y:np.ndarray, C:float, kernel:str, clfs:list) -> None:
    test_log = open(f'./Results/logs/test_svm_{kernel}_{C}_log.txt', 'w')
    print(f"test {kernel} svm with C={C}")

    accs = np.zeros(FOLDS)
    # Cross Validation
    for k in trange(FOLDS):
        test_x = dataset_X[k]
        test_y = dataset_Y[k]

        prediction = clfs[k].predict(test_x)
        accs[k] = accuracy_score(test_y, prediction)
    
    print('average accuracy: %.4f' % accs.mean(), file=test_log)
    print('average accuracy: %.4f' % accs.mean())
    print(accs, file=test_log)

    test_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', required=True, help='the regular constant', type=float)
    parser.add_argument('--kernel', required=True, help='the kernel function chosen', type=str)
    parser.add_argument('--save', action="store_true", help='whether to save the model parameters')
    parser.add_argument('--pretrained', action="store_true", help='whether have pretrained')
    args = parser.parse_args()

    EEG_X = sio.loadmat('./SEED_III/EEG_X.mat')['X'][0]     # 15 x 3394 x 310
    EEG_Y = sio.loadmat('./SEED_III/EEG_Y.mat')['Y'][0]     # 15 x 3394 x 1

    if not args.pretrained:
        clfs = train(EEG_X, EEG_Y, args.C, args.kernel, args.save)
    else:
        clfs = []
        for k in trange(FOLDS):
            model_file = open(f'./Results/models/svm_{args.kernel}_{args.C}_{k}.pickle', 'rb')
            clf = pickle.load(model_file)
            model_file.close()
            clfs.append(clf)
    
    test(EEG_X, EEG_Y, args.C, args.kernel, clfs)
