import os.path
from collections import OrderedDict
import flwr as fl
import numpy as np
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvgM, FedProx
from typing import Dict, List, Optional, Tuple, Union
import torch
from flwr.common import (
    EvaluateIns,
    FitRes,
    Parameters,
    Scalar,
    Metrics,
    Context,
    ndarrays_to_parameters,
)
from baselines.task import Net, get_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


agg_best_acc = None
cur_strategy = None
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global agg_best_acc
    global cur_strategy
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    cur_acc = sum(accuracies) / sum(examples)
    is_best = False

    if agg_best_acc is None or cur_acc > agg_best_acc:
        is_best = True
        agg_best_acc = cur_acc


    def save_mode(dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

        net = Net(config=cur_strategy.config).to(device)

        aggregated_ndarrays: List[np.ndarray] = cur_strategy.model_nd_params


        model_keys = [
            k
            for k in net.state_dict().keys()
        ]

        state_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(model_keys, aggregated_ndarrays)
        )

        # net.load_state_dict(state_dict,strict=False)
        net.load_state_dict(state_dict, strict=True)

        # Save the model
        torch.save(net.state_dict(), f"{dir}/model.pth")

    save_dir = cur_strategy.config["save_dir"]
    dir = os.path.join(save_dir, "models")
    dir_latest = os.path.join(save_dir, "latest")

    save_mode(dir_latest)
    if is_best:
        cur_strategy.no_update_rounds = 0
        save_mode(dir)
    else:
        cur_strategy.no_update_rounds += 1

    return {"accuracy": cur_acc}
def init_baseline_strategy(context: Context,config):
    # Read from config
    save_dir = config["save_dir"]
    mode_init_filename = config["init_weights"]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if mode_init_filename != "":
        mode_path = os.path.join(save_dir, mode_init_filename)
        global_state_dict = torch.load(mode_path, weights_only=False)
        body_ndarray = [val.cpu().numpy() for _,val in global_state_dict.items()]
        parameters = ndarrays_to_parameters(body_ndarray)
    else:
        # Initialize model parameters
        ndarrays = get_weights(Net(config))
        parameters = ndarrays_to_parameters(ndarrays)

    strategy = None
    if config["method"] == "FedAvgM":
        strategy = SaveModelStrategy(
            fraction_fit=config["fraction_fit"],
            fraction_evaluate=config["fraction_evaluate"],
            min_fit_clients=config["min_fit_clients"],
            min_available_clients=config["min_available_clients"],
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=parameters,
            server_momentum=config["server_momentum_beta"],
            config=config
        )
    elif config["method"] == "FedProx":
        strategy = SaveModelStrategyFoProx(
            fraction_fit=config["fraction_fit"],
            fraction_evaluate=config["fraction_evaluate"],
            min_fit_clients=config["min_fit_clients"],
            min_available_clients=config["min_available_clients"],
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=parameters,
            proximal_mu=config["proximal_mu"],
            config=config
        )

    global cur_strategy
    cur_strategy = strategy
    return strategy


class SaveModelStrategyFoProx(FedProx):
    last_global_acc = 0
    net = None
    model_nd_params = None
    no_update_rounds = 0
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.classifier = Net(config).to(device)

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if server_round % self.config["test_interval_rounds"] == 0:
            return super().configure_evaluate(server_round, parameters, client_manager)
        return []

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        early_stopping_step = self.config["server_early_stopping_step"]
        if early_stopping_step != 0 and self.no_update_rounds > early_stopping_step:
            print("early_stopping")
            raise ValueError(
                "early Stopping"
            )

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        agg_ndarray = fl.common.parameters_to_ndarrays(aggregated_parameters)
        self.model_nd_params = agg_ndarray

        return aggregated_parameters, aggregated_metrics




class SaveModelStrategy(FedAvgM):
    last_global_acc = 0
    net = None
    model_nd_params = None
    no_update_rounds = 0
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.classifier = Net(config).to(device)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        interval = self.config["test_interval_rounds"]
        if server_round % interval == 0:
            return super().configure_evaluate(server_round,parameters,client_manager)
        return []

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        early_stopping_step = self.config["server_early_stopping_step"]
        if early_stopping_step != 0 and self.no_update_rounds > early_stopping_step:
            print("early_stopping")
            raise ValueError(
                "early Stopping"
            )
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        agg_ndarray = fl.common.parameters_to_ndarrays(aggregated_parameters)
        self.model_nd_params = agg_ndarray
        return aggregated_parameters, aggregated_metrics

