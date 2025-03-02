"""fedsdc: A Flower / PyTorch app."""
import json
import os.path
import random
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from baselines.base_strategy import init_baseline_strategy
import fedsdc.fedsdc_strategy as feddsc
import torch
from flwr.common import (
    Context,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def init_config(run_config):
    final_config = {}

    for k,v in run_config.items():
        final_config[k] = v

    save_dir = final_config["save_dir"]

    if "sdc_diverse_head" in final_config.keys():
        is_use_diverse = final_config["sdc_diverse_head"]
        if is_use_diverse == 1:
            rho_start = float(final_config["sdc_rho_start"])
            rho_end = float(final_config["sdc_rho_end"])
            client_cnt = final_config["client_cnt"]
            fraction_fit = final_config["fraction_fit"]
            is_shuffle = final_config["sdc_shuffle"]
            head_cnt = client_cnt
            if is_shuffle == 1:
                head_cnt = int(client_cnt * fraction_fit)
            head_compress_ratio = [random.uniform(rho_start,rho_end) for _ in range(head_cnt)]
            final_config["sdc_head_compress_ratio"] = head_compress_ratio
            final_config["sdc_head_cnt"] = head_cnt


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    config_path = os.path.join(save_dir, "config.txt")
    with open(config_path, 'w') as file:
        json.dump(final_config, fp=file, indent=4)
    return final_config

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    run_config = init_config(context.run_config)
    num_rounds = run_config["num_server_rounds"]
    method = run_config["method"]
    if method == "FedSDC":
        strategy = feddsc.init_fedsdc_diversity_sample_strategy(context=context,config=run_config)
    else:
        strategy = init_baseline_strategy(context=context,config=run_config)

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
