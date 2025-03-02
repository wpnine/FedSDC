"""Client implementation - can call FedPep and FedAvg clients."""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union
from sklearn.metrics import f1_score

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader
from fedsdc.fedsdc_model import FedSDCModelManager
import os
import json

PROJECT_DIR = Path(__file__).parent.parent.absolute()

FEDERATED_DATASET = None



class BaseClient(NumPyClient):
    """Implementation of Federated Averaging (FedAvg) Client."""

    # pylint: disable=R0913
    def __init__(
        self,
        client_id: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        config,
    ):
        """Initialize client attributes.

        Args:
            client_id: The client ID.
            trainloader: Client train data loader.
            testloader: Client test data loader.
            config: dictionary containing the client configurations.
            model_manager_class: class to be used as the model manager.
            client_state_save_path: Path for saving model head parameters.
                (Just for FedRep). Defaults to "".
        """
        super().__init__()

        self.client_id = client_id
        self.testloader=testloader
        self.trainloader=trainloader
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_manager = FedSDCModelManager(
            client_id=self.client_id,
            config=config,
            trainloader=trainloader,
            testloader=testloader,
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters."""
        return self.model_manager.model.get_parameters()

    def set_parameters(
        self, parameters: List[np.ndarray], evaluate: bool = False, param_pos = None
    ) -> None:
        if param_pos is None:
            param_pos = self.client_id
        self.model_manager.model.set_parameters(parameters,param_pos)

    def perform_train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local training to the whole model.

        Returns
        -------
            Dict with the train metrics.
        """
        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()

        return self.model_manager.train()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ):
        """Train the provided parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
            Tuple containing the locally updated model parameters, \
                the number of examples used for training and \
                the training metrics.
        """
        isshuffle = self.config["sdc_shuffle"]
        if isshuffle == 0:
            target_pos = self.client_id
        else:
            target_pos = config["params_pos"]
        self.set_parameters(parameters, param_pos=target_pos)
        train_results = self.perform_train()

        # Update train history
        train_results["client_id"] = self.client_id
        train_results["params_pos"] = target_pos

        train_results["train_cls"] = json.dumps(self.trainloader.dataset.class_counter)
        train_results["test_cls"] = json.dumps(self.testloader.dataset.class_counter)

        return self.get_parameters(config), len(self.model_manager.trainloader.dataset), train_results





class FedSDCClient(BaseClient):
    """Implementation of Federated Personalization (FedRep) Client."""

    def perform_train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:

        retrun_data = super().perform_train()

        return retrun_data


    def evaluate(
        self, parameters=None, config=None
    ):

        body_keys = [
            k
            for k in self.model_manager.model.body.state_dict().keys()
        ]

        head_cnt = int(config["head_cnt"])

        body_length = len(body_keys)

        body_params = parameters[:body_length]
        body_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(body_keys, body_params)
        )
        self.model_manager.model.body.load_state_dict(body_dict,strict=True)
        self.model_manager.model.eval()

        start_index = body_length
        for i in range(head_cnt):
            h = self.model_manager.model.choice_heads[i]
            h_len = len(h.state_dict().keys())
            head_params = parameters[start_index: start_index + h_len]
            head_state_dict = OrderedDict(
                (k, torch.from_numpy(v)) for k, v in zip(h.state_dict().keys(), head_params)
            )
            start_index += h_len
            h.load_state_dict(head_state_dict)
            h.eval()

        testloader = self.testloader
        device = self.device
        allP_t1, allT_t1 = [], []

        for i, data in enumerate(testloader):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            features = self.model_manager.model.body(inputs)

            sum_result = None
            vote_t1_cnt = [{} for i in range(len(inputs))]
            for j in range(head_cnt):
                c = self.model_manager.model.choice_heads[j]
                temp = c(features)
                top_1_probabilities, top_1_indices = torch.topk(temp, 1)
                votes_t1 = top_1_indices.cpu().numpy()


                rank_level = [1.0, 0.8, 0.5]

                for k, rank in enumerate(votes_t1):
                    for m, v in enumerate(rank):
                        if v not in vote_t1_cnt[k].keys():
                            vote_t1_cnt[k][v] = rank_level[m]
                        else:
                            vote_t1_cnt[k][v] += rank_level[m]

                if sum_result is None:
                    sum_result = temp
                else:
                    sum_result = sum_result + temp

            result_predicts = sum_result
            final_result_t1 = result_predicts.argmax(1).cpu().numpy()

            for j, v_cnt in enumerate(vote_t1_cnt):
                bigest_cnt = 0
                bigest_index = None
                conflict = 0

                for k, v in v_cnt.items():
                    if bigest_index is None or bigest_cnt < v:
                        bigest_cnt = v
                        bigest_index = k
                        conflict = 0
                    elif v == bigest_cnt:
                        conflict += 1

                if conflict == 0:
                    final_result_t1[j] = bigest_index

            allP_t1.extend(final_result_t1)
            allT_t1.extend(labels.cpu().numpy())

        acc_t1 = f1_score(allT_t1, allP_t1, average='micro')
        save_dir = self.config["save_dir"]
        test_log_file = os.path.join(save_dir, "trace.txt")
        log_info = {
            "weights": "global"
        }
        log_info["val_acc_t1"] = acc_t1
        log_info["val_cnt"] = self.model_manager.test_dataset_size()

        global_test_log = json.dumps(log_info)
        with open(test_log_file, 'a+') as file:
            file.write(global_test_log + "\n")

        return (
            0.0,
            self.model_manager.test_dataset_size(),
            {"accuracy_t1":acc_t1},
        )


