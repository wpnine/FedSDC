import torch
import torch.nn as nn
import torchvision.models as models




class resnet18_gn(nn.Module):

    def __init__(self):
        super(resnet18_gn, self).__init__()
        basemodel = models.resnet18(weights=None)
        basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        basemodel.maxpool = torch.nn.Identity()
        basemodel.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        basemodel.layer1[0].bResNet18n1 = nn.GroupNorm(num_groups=2, num_channels=64)
        basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
        print("num_ftrs", num_ftrs)
        self.n_outputs = num_ftrs


    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        return h

