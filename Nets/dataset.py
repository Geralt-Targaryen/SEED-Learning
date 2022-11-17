from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len