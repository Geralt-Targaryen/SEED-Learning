import torch
from torch import nn
from torch.autograd import Function


class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


class DANN(nn.Module):
    def __init__(self, n0=310, n1=64, nco=3, ndo=15, dropout=0):
        super(DANN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n0, n1),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(n1, nco),
            nn.Sigmoid()
        )
        self.dclf = nn.Sequential(
            nn.Linear(n1, ndo),
            nn.Sigmoid()
        )

    def forward(self, x, alpha=0):
        encoded = self.encoder(x)
        classification = self.clf(encoded)
        domain = self.dclf(ReverseLayer.apply(encoded, alpha))
        return classification, domain, encoded

    def predict(self, x):
        encoded = self.encoder(x)
        classification = self.clf(encoded)
        return torch.argmax(classification, dim=1)


