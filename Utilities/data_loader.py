import torch.utils.data as data
import numpy as np

FOLDS = 15


# return 310d feature, scalar label and scalar domain
class GetLoader(data.Dataset):
    def __init__(self, dataset_X, dataset_Y, domain, transform=None):
        self.transform = transform

        if domain >= 0:
            self.features = dataset_X[domain]                               # 3394 x 310
            self.labels = dataset_Y[domain]                                 # 3394 x 1
            self.domains = [domain for _ in range(len(self.features))]      # 3394
        else:
            except_domain = -domain
            self.features = np.concatenate([dataset_X[(except_domain + i + 1) % FOLDS] for i in range(FOLDS-1)])
            self.labels = np.concatenate([dataset_Y[(except_domain + i + 1) % FOLDS] for i in range(FOLDS-1)])
            self.domains = np.concatenate([np.array([(except_domain + i + 1) % FOLDS for _ in range(len(self.features))]) for i in range(FOLDS-1)])
            # (3394x14) x 310
            # (3394x14) x 1
            # 3394x14
        self.labels = self.labels + 1       # deal with label problem
        
        self.data_len = len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]                                      # 310
        label = self.labels[index][0]                                       # scalar
        domain = self.domains[index]                                        # scalar

        if self.transform is not None:
            feature = self.transform(feature)

        return feature, label, domain

    def __len__(self):
        return self.data_len
