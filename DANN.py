import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio
import pickle
import time
import os
import argparse
from sklearn.metrics import accuracy_score
from multiprocessing import Pool

from Nets.dataset import SimpleDataset
from Nets.DANN_model1 import DANN


EEG_X = sio.loadmat('./SEED_III/EEG_X.mat')['X'][0]  # 15 x 3394 x 310
EEG_Y = sio.loadmat('./SEED_III/EEG_Y.mat')['Y'][0]  # 15 x 3394 x 1

FOLDS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def coral_loss(source, target):
    d = source.shape[1]
    ns, nt = source.shape[0], target.shape[0]

    # source covariance (feature_dim x feature_dim)
    tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm between source and target
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4*d*d)

    return loss


def train_single_net(args):
    fold, lr_feature, lr_clf, lr_domain, n_feature, dropout, epoch_clf, normalization, Epoch, batch_size, source = args

    source = (fold + source) % FOLDS
    # adapt from subject source to subject fold

    print(f'Source: {source}, target: {fold}, PID: {os.getpid()}')
    tic = time.time()

    if not normalization:
        X_train = torch.from_numpy(EEG_X[source]).float().to(DEVICE)
        X_test = torch.from_numpy(EEG_X[fold]).float().to(DEVICE)
    else:
        source_mean = EEG_X[source].mean(axis=0)
        X_train = torch.from_numpy(EEG_X[source] - source_mean).float().to(DEVICE)
        X_test = torch.from_numpy(EEG_X[fold] - source_mean).float().to(DEVICE)

    Y_train = torch.from_numpy(np.squeeze(EEG_Y[source] + 1).reshape((-1, 1))).long()
    Y_test = torch.from_numpy(np.squeeze(EEG_Y[fold] + 1).reshape((-1, 1))).long()

    Y_domain_train = torch.from_numpy(np.ones((EEG_X[fold].shape[0], 1)) * 0).long()
    Y_domain_test = torch.from_numpy(np.ones((EEG_X[fold].shape[0], 1)) * 1).long()

    train_dataset = SimpleDataset(X_train, torch.concat([Y_train, Y_domain_train], axis=1).to(DEVICE))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = SimpleDataset(X_test, torch.concat([Y_test, Y_domain_test], axis=1).to(DEVICE))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = DANN(n0=X_train.shape[1], n1=n_feature, nco=3, ndo=2, dropout=dropout).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': lr_feature, 'weight_decay': 1e-2},
        {'params': model.clf.parameters(), 'lr': lr_clf, 'weight_decay': 1e-2},
        {'params': model.dclf.parameters(), 'lr': lr_domain},
    ], lr=lr_feature)

    lr_f_lambda = lambda epoch: 0.995 ** epoch
    lr_c_lambda = lambda epoch: 0.995 ** epoch
    lr_d_lambda = lambda epoch: 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lr_f_lambda, lr_c_lambda, lr_d_lambda])

    criterion_domain = nn.CrossEntropyLoss()
    criterion_clf = nn.CrossEntropyLoss(ignore_index=3)
    criterion_feature = nn.MSELoss()

    for epoch in range(Epoch):

        data_train_iter = iter(train_dataloader)
        len_dataloader_train = len(train_dataloader)
        data_test_iter = iter(test_dataloader)
        len_dataloader_test = len(test_dataloader)

        model.train()
        Loss_clf = []
        Loss_feature = []
        Loss_domain_train = []
        Loss_domain_test = []

        for _ in range(len_dataloader_train):
            x_train, y_train = data_train_iter.next()
            x_train, y_train, y_d_train = x_train, y_train[:, 0], y_train[:, 1]
            x_test, y_test = data_test_iter.next()
            x_test, y_test, y_d_test = x_test, y_test[:, 0], y_test[:, 1]

            # alpha controls whether the domain loss is passed back to the feature extractor
            alpha = 0 if epoch < epoch_clf else -1

            out_train, domain_train, feature_train = model(x_train, alpha=alpha)
            out_test, domain_test, feature_test = model(x_test, alpha=alpha)

            loss_clf_train = criterion_clf(out_train, y_train)
            Loss_clf.append(float(loss_clf_train))

            loss_coral = coral_loss(feature_train, feature_test)

            feature_train_mean = feature_train.mean(dim=0)
            feature_test_mean = feature_test.mean(dim=0)
            loss_feature = criterion_feature(feature_train_mean, feature_test_mean)
            Loss_feature.append(float(loss_feature))
            loss_domain_train = criterion_domain(domain_train, y_d_train)
            Loss_domain_train.append(float(loss_domain_train))
            loss_domain_test = criterion_domain(domain_test, y_d_test)
            Loss_domain_test.append(float(loss_domain_test))
            loss = loss_clf_train + 0.6 * loss_domain_train + 0.5 * loss_domain_test + 0.3 * loss_feature + loss_coral

            # ideally, domain classification accuracy should converge to 0.5, and domain loss ln 2 (about 0.7)
            # a = accuracy_score(y_d.cpu(), torch.argmax(domain, dim=1).cpu().detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_feature = np.mean(Loss_feature) if len(Loss_feature) != 0 else 0
        loss_clf = np.mean(Loss_clf) if len(Loss_clf) != 0 else 0
        loss_domain_train = np.mean(Loss_domain_train) if len(Loss_domain_train) != 0 else 0
        loss_domain_test = np.mean(Loss_domain_test) if len(Loss_domain_test) != 0 else 0

        model.eval()
        prediction = model.predict(X_train).cpu()
        acc = accuracy_score(Y_train, prediction)
        test_prediction = model.predict(X_test).cpu()
        test_acc = accuracy_score(EEG_Y[fold] + 1, test_prediction)

        if epoch % 10 == 0:
            print(f'{source} -> {fold}, Epoch: {epoch} | ', end='')
            print('feature loss: %.4f  |  clf loss: %.4f  |  ' % (float(loss_feature), float(loss_clf)), end='')
            print('domain loss: (%.4f, %.4f)  |  ' % (float(loss_domain_train), float(loss_domain_test)), end='')
            print('acc: (%.4f, %.4f)' % (acc, test_acc))
            # print('domain acc: %f' % a)
            with open(f'pickles/CORAL_{epoch}_{fold}_{source}', 'wb') as f:
                pickle.dump(test_prediction.numpy(), f)

    toc = time.time()
    print(f'training time: {toc - tic}', end='\n\n')

    return model


def train_model(fold, lr_feature, lr_clf, lr_domain, n_feature, dropout, epoch_clf,
                normalization, initialization, Epoch, batch_size, num_process, save):
    # train DANN
    print('************************** Fold %d **************************' % fold)
    tic = time.time()

    prediction_all = []  # prediction of all 14 networks within one fold

    with Pool(num_process) as pool:
        models = list(pool.map(
            train_single_net,
            [(fold, lr_feature, lr_clf, lr_domain, n_feature, dropout,
              epoch_clf, normalization, Epoch, batch_size, k + 1,
              ) for k in range(FOLDS - 1)]
        ))

    for k in range(FOLDS - 1):
        source = k + 1
        if not normalization:
            pass
        else:
            source_mean = EEG_X[(fold + source) % FOLDS].mean(axis=0)

        model = models[k].eval()
        if not normalization:
            test_prediction = model.predict(torch.from_numpy(EEG_X[fold]).float().to(DEVICE)).cpu()
        else:
            test_prediction = model.predict(torch.from_numpy(EEG_X[fold] - source_mean).float().to(DEVICE)).cpu()
        prediction_all.append(test_prediction.reshape((-1, 1)))

        if save:
            model_file = open(f'Results/models/dann_{(fold + source) % FOLDS}to{fold}.pickle', 'wb')
            pickle.dump(model, model_file)
            model_file.close()

    prediction_all = np.concatenate(prediction_all, axis=1)
    # 3394 x 14
    prediction = [np.argmax(np.bincount(prediction_all[i, :])) for i in range(prediction_all.shape[0])]
    # 3394 x 1
    acc = accuracy_score(EEG_Y[fold] + 1, prediction)
    print('test accuracy of fold %d: %f' % (fold, acc))

    toc = time.time()
    print(f' Training time of fold {fold}: {toc-tic}', end='\n\n')

    return acc


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--lr_feature', help='learning rate for feature extractor', default=8e-5, type=float)
    parser.add_argument('--lr_clf', help='learning rate for classifier', default=1e-4, type=float)
    parser.add_argument('--lr_domain', help='learning rate for domain adversarial', default=5e-5, type=float)
    parser.add_argument('--n_feature', help='number of features extracted by feature extractor', default=64, type=int)
    parser.add_argument('--dropout', help='dropout rate', default=0, type=float)
    parser.add_argument('--epoch_clf', help='number of epochs for classifier training without gradient reversed',
                        default=10, type=int)
    parser.add_argument('--epochs', help='the number of epochs', default=800, type=int)
    parser.add_argument('--batch_size', help='batch size, for both train and test', default=512, type=int)
    parser.add_argument('--num_process', help='number of parallel processes', default=3, type=int)

    parser.add_argument('--normalization', action='store_true', help='whether to normalize the input data')
    parser.add_argument('--initialization', help='initialization for net, only for `kaiming` and `xavier`',
                        default=None, type=str)
    parser.add_argument('--save', action='store_true', help='whether to save the models')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    torch.multiprocessing.set_start_method('spawn')

    ACC = []
    for fold in range(FOLDS):
        torch.manual_seed(1)
        np.random.seed(1)

        acc = train_model(fold=fold, lr_feature=args.lr_feature, lr_clf=args.lr_clf, lr_domain=args.lr_domain,
                          n_feature=args.n_feature, dropout=args.dropout, epoch_clf=args.epoch_clf,
                          normalization=args.normalization, initialization=args.initialization,
                          Epoch=args.epochs, batch_size=args.batch_size, num_process=args.num_process, save=args.save)
        ACC.append(acc)

    print('Average accuracy of all folds: %f' % np.mean(ACC))
    print(ACC)
