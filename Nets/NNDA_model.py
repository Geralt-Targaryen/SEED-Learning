from torch import nn


class NNDA(nn.Module):
    def __init__(self, n0=310, n1=64, n2=8, n3=3, dropout=0.1):
        super(NNDA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n0, n1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n1, n2),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n2, n3),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.encoder(x)
        output = self.classifier(feature)
        return feature, output