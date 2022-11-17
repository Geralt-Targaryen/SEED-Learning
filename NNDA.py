from sklearn.metrics import accuracy_score
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
from Nets.NNDA_model import NNDA

FOLDS = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)

EEG_X = sio.loadmat('./SEED_III/EEG_X.mat')['X'][0]     # 15 x 3394 x 310
EEG_Y = sio.loadmat('./SEED_III/EEG_Y.mat')['Y'][0]     # 15 x 3394 x 1


def train_single_nn(args):
    EPOCH, lr, batch_size, k, normalization = args
    print(f'Fold {k} starts training, PID: {os.getpid()}')
    tic = time.time()

    if normalization:
        train_x = np.concatenate([
            EEG_X[(k + i + 1) % FOLDS] - EEG_X[(k + i + 1) % FOLDS].mean(axis=0) for i in range(FOLDS - 1)
        ]).astype(np.float32)
    else:
        train_x = np.concatenate([EEG_X[(k + i + 1) % FOLDS] for i in range(FOLDS - 1)]).astype(np.float32)

    train_y = np.ravel(np.concatenate([EEG_Y[(k + i + 1) % FOLDS] for i in range(FOLDS - 1)]) + 1).astype(np.int64)

    dataset = SimpleDataset(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = NNDA(n0=310, n1=128, n2=32, n3=3, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCH):
        Loss = []

        for step, (x, y) in enumerate(dataloader):
            encoded, prediction = model(x)

            loss = criterion(prediction, y)
            Loss.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = accuracy_score(y.cpu(), torch.argmax(prediction, dim=1).cpu())
        if epoch % 10 == 0:
            print(f'Fold: {k}, epoch: {epoch}, step: {step}, | training loss: %.4f, training acc: %.4f' % (np.mean(Loss), acc))

    toc = time.time()
    print(f'Training time of fold {k}: {toc - tic}')

    return model


def train(EPOCH: int, lr: float, batch_size: int,
          num_process: int, save: bool, normalization: bool) -> list:
    train_log = open(f'./Results/logs/train_nnda_log.txt', 'w')
    print(f"train neural networks for domain adaptation")

    tic = time.time()

    with Pool(num_process) as p:
        clfs = list(p.map(
            train_single_nn,
            [(EPOCH, lr, batch_size, k, normalization) for k in range(0, FOLDS)]
        ))

    toc = time.time()
    print('training time of all folds: %f' % (toc - tic), file=train_log)

    if save:
        for k in range(FOLDS):
            model_file = open(f'./Results/models/nn_{k}.pickle', 'wb')
            pickle.dump(clfs[k], model_file)
            model_file.close()

    train_log.close()
    return clfs


def test(clfs: list, normalizatoin: bool) -> None:
    test_log = open(f'./Results/logs/test_nnda_log.txt', 'w')
    print(f"test neural network domain adaptation")

    accs = np.zeros(FOLDS)
    for k in trange(FOLDS):
        if normalizatoin:
            test_x = torch.from_numpy(EEG_X[k] - EEG_X[k].mean(axis=0)).float().to(device)
        else:
            test_x = torch.from_numpy(EEG_X[k]).float().to(device)
        test_y = EEG_Y[k] + 1

        clfs[k].eval()
        prediction = clfs[k](test_x)[1]
        accs[k] = accuracy_score(test_y, torch.argmax(prediction, dim=1).cpu())

    print('average accuracy: %.4f' % accs.mean(), file=test_log)
    print('average accuracy: %.4f' % accs.mean())
    print(accs, file=test_log)
    print(accs)

    test_log.close()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-4, help='learning rate', type=float)
    parser.add_argument('--EPOCH', default=50, help='number of training epochs', type=int)
    parser.add_argument('--batch_size', default=512, help='batch size', type=int)
    parser.add_argument('--num_process', default=8, help='number of parallel processes', type=int)
    parser.add_argument('--save', action="store_true", help='whether to save the model parameters')
    parser.add_argument('--pretrained', action="store_true", help='whether have pretrained')
    parser.add_argument('--normalization', action="store_true", help='whether to normalize the input')
    args = parser.parse_args()

    if not args.pretrained:
        clfs = train(args.EPOCH, args.lr, args.batch_size, args.num_process, args.save, args.normalization)
    else:
        clfs = []
        for k in trange(FOLDS):
            with open(f'./Results/models/nn_{k}.pickle', 'rb') as f:
                clf = pickle.load(f)
                clfs.append(clf)

    test(clfs, args.normalization)
