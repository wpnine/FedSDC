"""fedsdc: A Flower / PyTorch app."""
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
from baselines.base_client import FlowerClient

def client_fn(context: Context):
    global client_map

    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    client_id = context.node_config["partition-id"]
    root_dir = context.run_config["save_dir"]
    config_path = os.path.join(root_dir,"config.txt")
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = config["local_batch_size"]


    save_dir = os.path.join(root_dir, "client_" + str(client_id))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    config["save_dir"] = save_dir

    data_dir = config["dataset"]
    with open(data_dir, 'r') as file:
       data = json.load(file)


    client_info = data[client_id]
    data_type = config["data_type"]
    if data_type == "ham":
        train_dataset,test_dataset = ham.init_ham_train(client_info)
    else:
        train_data = client_info["train_set"]
        test_data = client_info["test_set"]
        train_dataset = bds.init_with_obj(train_data,True)
        test_dataset = bds.init_with_obj(test_data,False)


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,shuffle=True,num_workers=8
    )
    test_loader = DataLoader(
        test_dataset , batch_size=batch_size, shuffle=False,num_workers=8
    )

    method = config["method"]
    client = None
    if method in ["FedAvgM", "FedProx"]:
        client = FlowerClient(train_loader, test_loader, config).to_client()
        client = client.to_client()
    elif method in ["FedSDC"]:
        client = FedSDCClient(
            client_id=client_id,
            trainloader=train_loader,
            testloader=test_loader,
            config=config,
        )
        client = client.to_client()

    return client


# Flower ClientApp
app = ClientApp(client_fn)
