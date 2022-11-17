import torch
from torch import nn
from torch.autograd import Function


class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha=-1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


class ADDA(nn.Module):
    def __init__(self, n0=310, n1=128, n2=32, nco=3, dropout=0):
        super(ADDA, self).__init__()
        self.extractor = nn.Sequential(
            nn.Linear(n0, n1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(n1, n2),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(n2, nco),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.extractor(x)
        prediction = self.predictor(feature)
        return feature, prediction

    def predict(self, x):
        return torch.argmax(self.forward(x)[1], dim=1)


class DCLF(nn.Module):
    def __init__(self, n0=64, n1=2):
        super(DCLF, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n0, n1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha=-1):
        prediction = self.classifier(ReverseLayer.apply(x, alpha))
        return prediction


