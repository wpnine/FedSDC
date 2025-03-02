import json
import os.path
import torch
import pandas as pd
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import dataset.blood_datasets as bds
from baselines.task import Net, get_weights, set_weights, test, train
from torch.utils.data import DataLoader
from fedsdc.fedsdc_client import FedSDCClient
import dataset.ham_dataset as ham

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader,config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net(config).to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

        self.save_dir = config["save_dir"]
        self.config = config


    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)

        proximal_mu = None
        if "proximal_mu" in config.keys():
            proximal_mu = float(config["proximal_mu"])

        results = train(
            self.net,
            self.trainloader,
            self.device,
            self.config,
            proximal_mu=proximal_mu,
        )

        results["train_cls"] = json.dumps(self.trainloader.dataset.class_counter)
        results["test_cls"] = json.dumps(self.valloader.dataset.class_counter)

        return get_weights(self.net), len(self.trainloader.dataset.sampleList), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        # loss, accuracy = test(self.net, self.valloader, self.device)
        val_cnt = len(self.valloader.dataset.sampleList)
        log_info = {
            "weights": "global"
        }
        test_log_file = os.path.join(self.save_dir, "trace.txt")

        val_loss, val_acc = test(self.net, self.valloader, self.device)
        log_info["val_loss"] = val_loss
        log_info["val_acc"] = val_acc
        log_info["val_cnt"] = val_cnt


        global_test_log = json.dumps(log_info)
        with open(test_log_file, 'a+') as file:
            file.write(global_test_log + "\n")

        return val_loss, len(self.valloader.dataset), {"accuracy": val_acc}

