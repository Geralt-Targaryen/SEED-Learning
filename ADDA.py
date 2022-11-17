import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
import time
import argparse
from multiprocessing import Pool
from sklearn.metrics import accuracy_score

from Nets.ADDA_model import ADDA, DCLF
from Nets.dataset import SimpleDataset

EEG_X = sio.loadmat('SEED_III/EEG_X.mat')['X'][0]    # 15 x 3394 x 310
EEG_Y = sio.loadmat('SEED_III/EEG_Y.mat')['Y'][0]    # 15 x 3394 x 1

FOLDS = 15
device = "cuda" if torch.cuda.is_available() else "cpu"


ACC = []


def train_single_net(args):
    torch.manual_seed(1)
    np.random.seed(1)

    fold, lr_feature, lr_clf, lr_domain, Epoch, Epoch_domain, batch_size, dropout, normalization, source = args
    source = (fold + source) % FOLDS

    print(f'Source: {source}, target: {fold}')
    tic = time.time()

    X = (EEG_X[source] - EEG_X[source].mean(axis=0)) if normalization else EEG_X[source]
    X_target = (EEG_X[fold] - EEG_X[fold].mean(axis=0)) if normalization else EEG_X[fold]

    X = torch.from_numpy(X).float().to(device)
    X_target = torch.from_numpy(X_target).float().to(device)

    Y = torch.from_numpy(np.squeeze(EEG_Y[source]).reshape((-1, 1)) + 1).long().to(device)
    # X: 3394 x 310
    # Y: 3394 x 1

    dataset = SimpleDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ADDA(n0=310, n1=64, n2=32, nco=3, dropout=dropout).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.extractor.parameters(), 'lr': lr_feature},
        {'params': model.predictor.parameters(), 'lr': lr_clf},
    ], lr=lr_feature)

    lr_f_lambda = lambda epoch: 0.998 ** epoch
    lr_c_lambda = lambda epoch: 0.998 ** epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lr_f_lambda, lr_c_lambda])

    criterion = nn.CrossEntropyLoss()

    print(f'{source} -> {fold}: training on source domain')

    for epoch in range(Epoch):
        model.train()
        Loss = []
        for step, (x, y) in enumerate(dataloader):
            y = y[:, 0]
            # x: batch x 310, y: batch x 1

            _, out = model(x)

            loss = criterion(out, y)
            Loss.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss = np.mean(Loss)

        model.eval()
        prediction = model.predict(X.to(device)).cpu()
        acc = accuracy_score(EEG_Y[source]+1, prediction)

        if epoch % 20 == 0:
            print(f'{source} -> {fold}, epoch: {epoch}  |  ', end='')
            print('clf loss: %.4f  |  ' % float(loss), end='')
            print('training acc: %.4f' % acc)

    ###############################################################################################
    ###############################################################################################

    model_target: ADDA = copy.deepcopy(model)
    model_domain = DCLF(n0=32, n1=2).to(device)

    # adversarial training
    optimizer = torch.optim.Adam([
        {'params': model.extractor.parameters(), 'lr': 0},
        {'params': model.predictor.parameters(), 'lr': 0},
        {'params': model_target.extractor.parameters(), 'lr': lr_feature},
        {'params': model_target.predictor.parameters(), 'lr': 0},
        {'params': model_domain.classifier.parameters(), 'lr': lr_domain},
    ], lr=0)

    lr_0_lambda = lambda epoch: 0
    lr_f_lambda = lambda epoch: 0.03 * 0.99 ** epoch
    lr_d_lambda = lambda epoch: 0.99 ** epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lr_0_lambda, lr_0_lambda, lr_f_lambda, lr_0_lambda,
                                                             lr_d_lambda])
    criterion = nn.CrossEntropyLoss()
    criterion_f = nn.MSELoss()

    Y_domain = torch.from_numpy(np.ravel(np.ones((EEG_X[fold].shape[0], 1)) * 0)).long().to(device)
    Y_domain_target = torch.from_numpy(np.ravel(np.ones((EEG_X[fold].shape[0], 1)) * 1)).long().to(device)
    Y_d_cpu = np.ravel(np.ones((EEG_X[fold].shape[0], 1)) * 0)
    Y_d_t_cpu = np.ravel(np.ones((EEG_X[fold].shape[0], 1)) * 1)

    train_dataset = SimpleDataset(X, Y_domain)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = SimpleDataset(X_target, Y_domain_target)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f'{source} -> {fold}: adapting to target domain')
    model.train()

    for epoch in range(Epoch_domain):
        model_target.train()
        model_domain.train()

        data_train_iter = iter(train_dataloader)
        len_dataloader_train = len(train_dataloader)
        data_test_iter = iter(test_dataloader)

        Loss_domain_source = []
        Loss_domain_target = []
        Loss_feature = []

        for _ in range(len_dataloader_train):
            x, y_d = data_train_iter.next()
            # x, y_d = x.to(device), y_d.to(device)
            x_target, y_d_target = data_test_iter.next()
            # x_target, y_d_target = x_target.to(device), y_d_target.to(device)

            f_source, _ = model(x)
            f_target, _ = model_target(x_target)
            out_source = model_domain(f_source, alpha=-1)
            out_target = model_domain(f_target, alpha=-1)

            loss_source = criterion(out_source, y_d)
            loss_target = criterion(out_target, y_d_target)
            loss_feature = criterion_f(f_source.mean(dim=0), f_target.mean(dim=0))
            Loss_domain_source.append(float(loss_source))
            Loss_domain_target.append(float(loss_target))
            Loss_feature.append(float(loss_feature))

            loss = 0.55 * loss_source + 0.5 * loss_target + 0.4 * loss_feature

            loss.backward()
            optimizer.step()
        scheduler.step()
        model_domain.eval()
        model_target.eval()

        f_source = model(X.to(device))[0]
        f_target, prediction = model_target(X_target.to(device))
        out_source = torch.argmax(model_domain(f_source), dim=1).cpu()
        out_target = torch.argmax(model_domain(f_target), dim=1).cpu()
        acc_source = accuracy_score(Y_d_cpu, out_source)
        acc_target = accuracy_score(Y_d_t_cpu, out_target)

        prediction = torch.argmax(prediction, dim=1).cpu()
        acc = accuracy_score(EEG_Y[fold] + 1, prediction)

        if epoch % 20 == 0:
            print(f'{source} -> {fold}, epoch : {epoch} | domain loss: (%.3f, %.3f), | domain acc: (%.2f, %.2f) | '
                  % (np.mean(Loss_domain_source), np.mean(Loss_domain_target), acc_source, acc_target), end='')
            print(f'f loss: %.3f | test acc: %.4f'
                  % (np.mean(Loss_feature), acc))

    toc = time.time()
    print(f'{source} -> {fold}, training time: {toc - tic}')

    return model_target


def train_model(fold, lr_feature, lr_clf, lr_domain, Epoch, Epoch_domain, batch_size, dropout, normalization, num_process):
    global ACC
    # train ADDA
    print('************************** Fold %d **************************' % fold)
    tic = time.time()

    prediction_all = []     # prediction of all 14 networks within one fold
    models = []

    n = (FOLDS - 1) // num_process

    for i in range(n + 1 if n*num_process != (FOLDS-1) else n):

        with Pool(num_process) as pool:
            models += list(pool.map(
                train_single_net,
                [(fold, lr_feature, lr_clf, lr_domain, Epoch, Epoch_domain,
                  batch_size, dropout, normalization,  k + 1,
                  ) for k in range(i * num_process, min((i + 1) * num_process, FOLDS - 1))]
            ))

        for k in range(i * num_process, min((i + 1) * num_process, FOLDS - 1)):
            if not normalization:
                mean = 0
            else:
                mean = EEG_X[fold].mean(axis=0)

            model = models[k].eval()
            test_prediction = model.predict(torch.from_numpy(EEG_X[fold] - mean).float().to(device)).cpu()
            prediction_all.append(test_prediction.reshape((-1, 1)))

    prediction_all = np.concatenate(prediction_all, axis=1)
    # 3394 x 14
    prediction = [np.argmax(np.bincount(prediction_all[i,:])) for i in range(prediction_all.shape[0])]
    # 3394 x 1
    acc = accuracy_score(EEG_Y[fold] + 1, prediction)
    print('test accuracy of fold %d: %f' % (fold, acc))
    ACC.append(acc)

    toc = time.time()
    print(f'Training time of fold {fold}: {toc-tic}', end='\n\n')


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--lr_feature', help='learning rate for feature extractor', default=5e-5, type=float)
    parser.add_argument('--lr_clf', help='learning rate for classifier', default=5e-5, type=float)
    parser.add_argument('--lr_domain', help='learning rate for domain adversarial', default=5e-4, type=float)
    parser.add_argument('--dropout', help='dropout rate', default=0, type=float)
    parser.add_argument('--epoch_domain', help='number of epochs for classifier training with gradient reversed',
                        default=500, type=int)
    parser.add_argument('--epochs', help='the number of epochs', default=500, type=int)
    parser.add_argument('--batch_size', help='batch size, for both train and test', default=512, type=int)
    parser.add_argument('--num_process', help='number of parallel processes', default=7, type=int)

    parser.add_argument('--normalization', action='store_true', help='whether to normalize the input data')

    parser.add_argument('--save', action='store_true', help='whether to save the models')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    torch.multiprocessing.set_start_method('spawn')

    for fold in range(FOLDS):
        torch.manual_seed(1)
        np.random.seed(1)
        train_model(fold=fold, lr_feature=args.lr_feature, lr_clf=args.lr_clf, lr_domain=args.lr_domain,
                    Epoch=args.epochs, Epoch_domain=args.epoch_domain, batch_size=args.batch_size,
                    dropout=args.dropout, normalization=args.normalization, num_process=args.num_process)

    print('Average accuracy of all folds: %f' % np.mean(ACC))