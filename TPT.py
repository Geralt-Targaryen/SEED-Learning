from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn import svm

import pickle
import scipy.io as sio
import numpy as np
import time
import os
import argparse
from tqdm import trange
from multiprocessing import Pool

import torch
from torch import nn
from torch.utils.data import DataLoader
from Nets.dataset import SimpleDataset
from Nets.TPT_model import Model

FOLDS = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)

EEG_X = sio.loadmat('./SEED_III/EEG_X.mat')['X'][0]     # 15 x 3394 x 310
EEG_Y = sio.loadmat('./SEED_III/EEG_Y.mat')['Y'][0]     # 15 x 3394 x 1


def train_single_nn(args):

    train_x, train_y, EPOCH, lr, batch_size, k, fold = args
    print(f'{k} -> {fold}, PID: {os.getpid()}')
    tic = time.time()

    dataset = SimpleDataset(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = Model(n0=32, n1=16, n2=3, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCH):
        Loss = []
        for step, (x, y) in enumerate(dataloader):
            prediction = model(x)

            loss = criterion(prediction, y)
            Loss.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = accuracy_score(y.cpu(), torch.argmax(prediction, dim=1).cpu())
        if epoch % 20 == 0:
            print(f'{k} -> {fold}, epoch: {epoch}, step: {step}, | training loss: %.4f, training acc: %.4f' % (
                np.mean(Loss), acc))
    toc = time.time()
    print(f'{k} -> {fold}, training time: {toc - tic}')

    return model


def train(EPOCH: int, lr: float, batch_size: int, num_process: int, save: bool,
          normalization: bool, pretrained: bool, fold: int):

    print(f"Adapting to target subject {fold}, training source classifiers")

    tic = time.time()

    # apply dimensionality reduction
    EEG_X_reduced = []
    with open(f'./Results/models/nn_{fold}.pickle', 'rb') as f:
        encoder = pickle.load(f).eval()
    for k in range(FOLDS):
        EEG_X_reduced.append(
            encoder(torch.from_numpy(
                EEG_X[k] - (EEG_X[k].mean(axis=0) if normalization else 0)
            ).float().to(device))[0].cpu().detach().numpy()
        )

    if pretrained:
        with open(f'./Results/models/nn_tpt_{fold}.pickle', 'rb') as f:
            clfs = pickle.load(f)
    else:
        train_log = open(f'./Results/logs/train_tpt_source_log.txt', 'w')

        with Pool(num_process) as pool:
            clfs = list(pool.map(
                train_single_nn,
                [(EEG_X_reduced[(fold + k + 1) % FOLDS].astype(np.float32), np.ravel(EEG_Y[(fold + k + 1) % FOLDS] + 1).astype(np.int64),
                  EPOCH, lr, batch_size, (fold + k + 1) % FOLDS, fold) for k in range(0, FOLDS - 1)]
            ))

        # clfs: [clf(fold+1), clf(fold+2), ..., clf(fold+14)]
        if save:
            with open(f'./Results/models/nn_tpt_{fold}.pickle', 'wb') as f:
                pickle.dump(clfs, f)


    toc = time.time()
    print(f'training time of fold {fold}: %f' % (toc - tic), file=train_log)
    print(f'training time of fold {fold}: %f' % (toc - tic))
    train_log.close()
    clf = transfer(EEG_X_reduced, clfs, kernel='rbf', C=1, fold=fold)

    acc = test(EEG_X_reduced[fold].astype(np.float32), EEG_Y[fold] + 1, clf)
    print(f'Test accuracy of fold {fold}: {acc}')
    return acc


def transfer(dataset_X: np.ndarray, clfs: list, kernel: str, C, fold):
    print(f'Transfer to target subject {fold}')
    parameters = []
    for k in range(FOLDS - 1):
        parameters.append(clfs[k].extract_parameter())
    # parameters[i]: 1 x 323

    features = []
    for k in range(FOLDS - 1):
        features.append(np.concatenate([
            dataset_X[(fold + k + 1) % FOLDS][:, i] for i in range(dataset_X[(fold + k + 1) % FOLDS].shape[1])
        ]).reshape((1, -1)))
    # features[i]: 1 x 54304 (3394*16)

    train_x = np.concatenate(features)
    train_y = np.concatenate(parameters)
    print(train_x.shape)    # 14 x 108608
    print(train_y.shape)    # 14 x 579

    transfer_model = MultiOutputRegressor(svm.SVR(kernel=kernel, C=C)).fit(train_x, train_y)
    target_param = transfer_model.predict(features[k]).squeeze()

    model = Model(n0=32, n1=16, n2=3, dropout=0.1)
    model.inject_parameter(target_param)
    model.to(device)

    return model


def test(test_x, test_y, clf):

    test_x = torch.from_numpy(test_x).to(device)

    clf.eval()
    prediction = clf(test_x)
    acc = accuracy_score(test_y, torch.argmax(prediction, dim=1).cpu())

    return acc


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--lr', default=3e-3, help='learning rate', type=float)
    parser.add_argument('--EPOCH', default=400, help='number of training epochs', type=float)
    parser.add_argument('--batch_size', default=128, help='batch size', type=int)
    parser.add_argument('--num_process', default=3, help='number of parallel processes', type=int)
    parser.add_argument('--save', action="store_true", help='whether to save the model parameters')
    parser.add_argument('--pretrained', action="store_true", help='whether have pretrained')
    parser.add_argument('--pretrained_encoder', action="store_true",
                        help='whether have pretrained encoder (i.e. NNDA)')
    parser.add_argument('--normalization', action="store_true", help='whether to normalize the input')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    torch.multiprocessing.set_start_method('spawn')

    if not args.pretrained_encoder:
        os.system(f'python NNDA.py --save {"--normalization" if args.normalization else ""}')

    # train classifiers on each source subject
    ACC = []
    tic = time.time()
    for k in range(FOLDS):
        ACC.append(train(args.EPOCH, args.lr, args.batch_size, args.num_process, args.save,
                          args.normalization, args.pretrained, k))

    print(f'Training time of all folds: {time.time() - tic}')
    print(f'Average accuracy of all folds: {np.mean(ACC)}')

