import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models






class resnet50(nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self):
        super(resnet50, self).__init__()
        self.network = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.n_outputs = 2048

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))
        # return self.network(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
