"""Abstract class for splitting a model into body and head."""

from abc import ABC
from typing import Dict, List, Union
# import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from backbone.resnet50 import resnet50
from backbone.resnet18_gn import resnet18_gn


# pylint: disable=R0902, R0913, R0801
class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        config,
        trainloader: DataLoader,
        testloader: DataLoader,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            trainloader: Client train dataloader.
            testloader: Client test dataloader.
            client_save_path: Path to save the client model head state.
            model_split_class: Class to be used to split the model into body and head \
                (concrete implementation of ModelSplit).
        """
        super().__init__()
        self.config = config
        self.client_id = client_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model: Classifer = self._create_model()

    def _create_model(self) -> nn.Module:
        return Classifer(self.config)

    def test_dataset_size(self):
        return len(self.testloader.dataset.sampleList)
    @property
    def model(self):
        """Return model."""
        return self._model

    def train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Returns
        -------
            Dict containing the train metrics.
        """
        # Load client state (head) if client_save_path is not None and it is not empty
        # if self.client_save_path is not None and os.path.isfile(self.client_save_path):
        #     self._model.head.load_state_dict(torch.load(self.client_save_path))
        config = self.config
        num_local_epochs = config["local_epoch"]
        train_mode = config["local_train_mode"]
        local_second_epoch = 0
        if train_mode in [2,3]:
            local_second_epoch = config["local_second_epoch"]


        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=config["local_lr"],
            weight_decay=config["local_wd"],
            momentum=config["local_momentum"],
        )
        loss: torch.Tensor = 0.0

        self._model.train()
        for i in range(num_local_epochs + local_second_epoch):

            if train_mode == 1:
                self._model.enable_body()
                self._model.enable_head()
            elif train_mode == 2:
                if i < num_local_epochs:
                    self._model.enable_body()
                    self._model.disable_head()
                else:
                    self._model.disable_body()
                    self._model.enable_head()
            elif train_mode == 3:
                if i < num_local_epochs:
                    self._model.disable_body()
                    self._model.enable_head()
                else:
                    self._model.enable_body()
                    self._model.disable_head()


            # correct, total = 0, 0
            for batch in self.trainloader:
                images = batch[0]
                labels = batch[1]

                outputs = self._model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # total += labels.size(0)
                # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        return {"loss": float(loss.item()), "accuracy": float(0)}





class Classifer(nn.Module):
    """CNN model for CIFAR100 dataset.

    Refer to
    https://github.com/rahulv0205/fedrep_experiments/blob/main/models/Nets.py
    """

    def __init__(self,config):
        """Initialize the model."""
        super().__init__()
        # Note that in the official implementation, the body has no BN layers.
        # However, no BN will definitely lead training to collapse.
        self.config = config
        backbone = config["backbone"]
        if backbone == "ResNet50":
            self.body = resnet50()
        elif backbone == "ResNet18":
            self.body = resnet18_gn()

        class_num = config["class_num"]

        self.choice_heads = []

        head_ratio = config["sdc_head_compress_ratio"]
        head_mode = config["sdc_diverse_head"]
        head_dropout = config["sdc_head_dropout"]
        head_cnt = config["sdc_head_cnt"]

        for i in range(head_cnt):
            if head_mode == 1:
                convert_cnt = int(head_ratio[i] * self.body.n_outputs)
                l1 = torch.nn.Linear(self.body.n_outputs, convert_cnt)
                l2 = torch.nn.Dropout(head_dropout)
                l3 = torch.nn.Linear(convert_cnt, class_num)
                head1 = torch.nn.Sequential(l1, l2, l3)
                self.choice_heads.append(head1)
            else:
                l1 = torch.nn.Linear(self.body.n_outputs, class_num)
                self.choice_heads.append(l1)

        self.head = None



    def forward(self,x):
        return self.head(self.body(x))

    def init_adapter(self):
        self.body.init_adapter()

    def use_head(self,head_no):
        self.head_no = head_no
        self.head = self.choice_heads[head_no]

    def is_no_contains(self, key, filter):
        for _, f in enumerate(filter):
            if key.startswith(f):
                return False
        return True
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters.

        Returns
        -------
            Body and head parameters
        """
        global_params = [
            val.cpu().numpy()
            for key, val in self.state_dict().items()
            if key.startswith("body")
        ]

        personal = [
            val.cpu().numpy()
            for key, val in self.state_dict().items()
            if key.startswith("head")
        ]

        return global_params + personal

    def set_parameters(self, parameters, params_pos) -> None:
        self.use_head(params_pos)
        global_keys = [
            k
            for k in self.state_dict().keys()
            if k.startswith("body")
        ]

        head_keys = [
            k
            for k in self.state_dict().keys()
            if k.startswith("head")
        ]
        all_ndarray = parameters
        personal_start_index = len(global_keys)
        personal_end_index = None
        for i,h in enumerate(self.choice_heads):
            if i < params_pos:
                personal_start_index += len(h.state_dict().keys())
            else:
                personal_end_index = personal_start_index + len(h.state_dict().keys())

        global_ndarray = all_ndarray[:len(global_keys)]
        personal_ndarray = all_ndarray[personal_start_index: personal_end_index]

        global_ndarray.extend(personal_ndarray)
        global_keys.extend(head_keys)
        state_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(global_keys, global_ndarray)
        )

        if len(personal_ndarray) == 0:
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=True)

    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for key,param in self.named_parameters():
            if key.startswith("head"):
                param.requires_grad = True



    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""

        # for key,param in self.body.named_parameters():
        for key,param in self.named_parameters():
            if key.startswith("body"):
                param.requires_grad = True



    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for key,param in self.named_parameters():
            if key.startswith("head"):
                param.requires_grad = False


    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        # for key,param in self.body.named_parameters():
        for key,param in self.named_parameters():
            if key.startswith("body"):
                param.requires_grad = False


class FedSDCModelManager(ModelManager):
    def __init__(self, **kwargs):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
        """
        super().__init__( **kwargs)

    def _create_model(self) -> Classifer:
        classifier = Classifer(self.config).to(self.device)
        for i,h in enumerate(classifier.choice_heads):
            h.to(self.device)
        return classifier
