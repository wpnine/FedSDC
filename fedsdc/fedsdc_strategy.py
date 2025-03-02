import json
import os.path
import random
from collections import OrderedDict
from typing import List, Tuple, Optional
import flwr as fl
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, ClientManager
from flwr.server.client_proxy import ClientProxy
from fedsdc.fedsdc_model import Classifer
from flwr.server.strategy import FedAvg,FedAvgM
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.init as init

from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Metrics,
    Context,
    ndarrays_to_parameters,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agg_best_acc_t1 = None
cur_strategy = None
# Define metric aggregation function
def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global agg_best_acc_t1
    global cur_strategy
    # Multiply accuracy of each client by number of examples used
    accuracies_t1 = [num_examples * m["accuracy_t1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    cur_acc_t1 = sum(accuracies_t1) / sum(examples)
    is_best_t1 = False

    if agg_best_acc_t1 is None or cur_acc_t1 > agg_best_acc_t1:
        is_best_t1 = True
        agg_best_acc_t1 = cur_acc_t1

    def save_mode(dir):
        net = Classifer(cur_strategy.config).to(device)
        global_keys = net.body.state_dict().keys()
        state_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(global_keys, cur_strategy.body_nd_params)
        )

        if not os.path.isdir(dir):
            os.makedirs(dir)

        # Save the model
        torch.save(state_dict, f"{dir}/global.pth")

        for i, h in enumerate(cur_strategy.headers):
            personal_keys = net.choice_heads[i].state_dict().keys()
            personal_dict = OrderedDict(
                (k, torch.from_numpy(v)) for k, v in zip(personal_keys, h)
            )
            torch.save(personal_dict, f"{dir}/personal_{i}.pth")
    save_dir = cur_strategy.config["save_dir"]
    dir_t1 = os.path.join(save_dir, "models_t1")
    dir_latest = os.path.join(save_dir, "latest")
    save_mode(dir_latest)

    if is_best_t1:
        cur_strategy.no_update_rounds = 0
        save_mode(dir_t1)
    else:
        cur_strategy.no_update_rounds += 1

    return {"accuracy": cur_acc_t1}
def init_fedsdc_diversity_sample_strategy(context: Context, config):
    # Read from config
    save_dir = config["save_dir"]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    all_head_params = []

    mode_init_dir = config["init_weights"]
    parameters = None
    if mode_init_dir != "":
        head_cnt = config["sdc_head_cnt"]
        mode_dir = os.path.join(save_dir, mode_init_dir)
        global_state_dict = torch.load(os.path.join(mode_dir, "global.pth"), weights_only=False)
        body_ndarray = [val.cpu().numpy() for _,val in global_state_dict.items()]
        heads = []
        for i in range(head_cnt):
            personal_pth = os.path.join(mode_dir,f"personal_{i}.pth")
            if not os.path.isfile(personal_pth):
                break
            personal_state_dict = torch.load(personal_pth, weights_only=False)
            personal_ndarray = [val.cpu().numpy() for _,val in personal_state_dict.items()]
            heads.append(personal_ndarray)
            all_head_params.append(personal_ndarray)
            body_ndarray += personal_ndarray
            parameters = ndarrays_to_parameters(body_ndarray)

    else:
        net = Classifer(config)
        body_keys = [
            k
            for k in net.state_dict().keys()
            if k.startswith("body")
        ]
        body_ndarray = []
        for key, val in net.state_dict().items():
            if key in body_keys:
                body_ndarray.append(val.cpu().numpy())


        if config["sdc_shuffle"] == 0:
            for h in range(len(net.choice_heads)):
                head = net.choice_heads[h]
                if isinstance(head,torch.nn.Linear):
                    init.xavier_normal_(head.weight)
                    init.constant_(head.bias, 0)
                else:
                    for i,head_item in enumerate(head):
                        if isinstance(head_item, torch.nn.Linear):
                            init.xavier_normal_(head_item.weight)
                            init.constant_(head_item.bias, 0)

                personal_ndarray = [val.cpu().numpy() for _, val in head.state_dict().items()]
                all_head_params.append(personal_ndarray)
                body_ndarray += personal_ndarray

        parameters = ndarrays_to_parameters(body_ndarray)


    global cur_strategy
    strategy = FedSDCStrategy(
        fraction_fit=config["fraction_fit"],
        fraction_evaluate=config["fraction_evaluate"],
        min_fit_clients=config["min_fit_clients"],
        min_available_clients=config["min_available_clients"],
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=parameters,
        server_momentum=config["server_momentum_beta"],
        config=config,
    )
    strategy.update_headers()
    if len(all_head_params) > 0:
        strategy.headers = all_head_params

    cur_strategy = strategy

    return strategy

class FedSDCStrategy(FedAvgM):
    classifier = None
    cacheHeadList = {}
    no_update_rounds = 0
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.classifier = Classifer(config).to(device)

    def update_headers(self):
        head_cnt = self.config["sdc_head_cnt"]
        self.headers = [None] * head_cnt
        self.head_update_vectors = [None] * head_cnt


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        interval = self.config["test_interval_rounds"]
        if server_round % interval == 0:
            result = super().configure_evaluate(server_round,parameters,client_manager)
            for i,r in enumerate(result):
                r[1].config["head_cnt"] = len(self.headers)
            return result
        return []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        early_stopping_step = self.config["server_early_stopping_step"]
        if early_stopping_step != 0 and self.no_update_rounds > early_stopping_step:
            print("early_stopping")
            raise ValueError(
                "early Stopping"
            )

        result = super().configure_fit(server_round,parameters,client_manager)
        random.shuffle(result)
        temp = {}
        for i,v in enumerate(result):
            config = FitIns(parameters, {
                "params_pos": i
            })
            result[i] = (v[0], config)

        return result


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        global_keys = [
            k
            for k in self.classifier.state_dict().keys()
            if k.startswith("body")
        ]

        is_use_rethink = False
        for i,r in enumerate(results):
            params = fl.common.parameters_to_ndarrays(r[1].parameters)
            params_pos = r[1].metrics["params_pos"]
            head = self.classifier.choice_heads[params_pos]
            head_keys = [k for k in head.state_dict().keys()]
            model_keys = global_keys + head_keys
            global_params = []
            personal_params = []

            for k,v in zip(model_keys,params):
                if k.startswith("body"):
                    global_params.append(v)
                else:
                    personal_params.append(v)

            self.headers[params_pos] = personal_params

            flparams = fl.common.ndarrays_to_parameters(global_params)
            r[1].parameters = flparams

        # Call aggregate_fit from base class to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        agg_ndarray = fl.common.parameters_to_ndarrays(aggregated_parameters)
        self.body_nd_params = agg_ndarray
        all_ndarray = []
        all_ndarray.extend(agg_ndarray)

        for _, h in enumerate(self.headers):
            all_ndarray.extend(h)
        aggregated_parameters = fl.common.ndarrays_to_parameters(all_ndarray)

        return aggregated_parameters, {}

