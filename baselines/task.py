"""fedsdc: A Flower / PyTorch app."""
import os.path
from collections import OrderedDict
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset.blood_datasets as bds
import torchvision
from backbone.resnet18_gn import resnet18_gn
from backbone.resnet50 import resnet50



class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self,config):
        super(Net, self).__init__()
        self.config = config
        backbone = config["backbone"]
        if backbone == "ResNet18":
            self.featurizer = resnet18_gn()
        else:
            self.featurizer = resnet50()

        class_num = config["class_num"]

        self.classifier = torch.nn.Linear(self.featurizer.n_outputs, class_num)

    def forward(self, x):
        f = self.featurizer(x)
        return self.classifier(f)



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    # print(net.state_dict().keys())
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


fds = None  # Cache FederatedDataset



def load_data(partition_id: int, num_partitions: int, batch_size: int, db_name: str, save_dir: str):
    cache_path = os.path.join(save_dir, "datasets.txt")
    if os.path.isfile(cache_path):
        print("load cache data:", db_name, partition_id)
        with open(cache_path, 'r') as file:
            data = json.load(file)
            train, val, test = bds.init_blood_datasets_with_objs(data, augment=True)
            train.env = db_name
            val.env = db_name
            test.env = db_name
    else:
        train, val, test = bds.init_blood_datasets("/home/wp/datasets", db_name, augment=True)
        train_json = json.dumps({
            "train": train.sampleList,
            "val": val.sampleList,
            "test": test.sampleList,
        })
        with open(cache_path, 'w') as file:
            file.write(train_json)

    train.env = db_name
    val.env = db_name
    test.env = db_name

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,num_workers=8
    )
    val_dataloader = DataLoader(val, batch_size=batch_size,num_workers=8)
    test_dataloader = DataLoader(val, batch_size=batch_size,num_workers=8)
    return train_loader, val_dataloader, test_dataloader



def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        shuffle = torch.randperm(len(xj))
        xj, yj = xj[shuffle], yj[shuffle]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def train(net, trainloader, device, config,proximal_mu = None):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    net.eval()
    epochs = config["local_epoch"]
    lr = config["local_lr"]
    wd = config["local_wd"]
    momentum = config["local_momentum"]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,weight_decay=wd,momentum=momentum)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01,weight_decay=0.1,momentum=0.5) #fedopt
    if proximal_mu is not None:
        global_params = [val.detach().clone() for val in net.parameters()]

    # is_mixup = True
    class_num = config["class_num"]

    net.train()


    for i in range(epochs):

        for i,data in enumerate(trainloader):

            images = data[0]
            labels = data[1]
            labels = (F.one_hot(labels, class_num) * 1.0).to(device)

            optimizer.zero_grad()
            if proximal_mu is not None:
                # print("do prox")
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += torch.square((local_weights - global_weights).norm(2))
                loss = F.cross_entropy(net(images.to(device)), labels) + (proximal_mu / 2) * proximal_term
            else:
                predicts = net(images.to(device))
                loss = F.cross_entropy(predicts, labels.to(device))

            loss.backward()
            optimizer.step()

    #
    # results = {
    #     "loss": loss.item(),
    #     "accuracy": 0,
    # }
    results = {
        "loss": 0.0,
        "accuracy": 0,
    }
    return results



def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for x,y in testloader:
            images = x.to(device)
            labels = y.to(device)
            outputs = net(images)

            # predict = torch.zeros(labels.size(0),0).to(device)
            # for _,v in enumerate(outputs):
            #     predict = torch.cat((predict,v),dim=1)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

