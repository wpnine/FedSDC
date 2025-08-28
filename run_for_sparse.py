import math
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import dataset.blood_datasets as bds
import dataset.ham_dataset as ham

from sklearn.metrics import cohen_kappa_score as kappa
from sklearn import metrics

from sklearn.metrics import balanced_accuracy_score as bal_acc
from sklearn.metrics import matthews_corrcoef, f1_score
from fedsdc.fedsdc_model import Classifer
import json
import math

def setup_dataloader(dataset_dic):
    batch_size = 32
    dl = DataLoader(dataset_dic,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4)
    return dl



def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)



def evaluate_multi_cls(y_true, y_pred):
    # preds
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    k = kappa(y_true, y_pred, weights='quadratic')
    bacc = bal_acc(y_true, y_pred)
    present_classes, _ = np.unique(y_true, return_counts=True)
    return k, mcc, f1, bacc


def calcResult(model,device,dataloader,classifiers,evaluate):
    allP, allT = [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            features = model.body(inputs)

            if isinstance(classifiers, list):
                sum_result = None
                vote_cnt = [{} for i in range(len(inputs))]
                for j,c in enumerate(classifiers):
                    temp = c(features)

                    top_n = 1
                    top_n_probabilities, top_n_indices = torch.topk(temp, top_n)
                    votes = top_n_indices.cpu().numpy()
                    rank_level = [1.0,0.8,0.5]
                    for k,rank in enumerate(votes):
                        for m, v in enumerate(rank):
                            if v not in vote_cnt[k].keys():
                                vote_cnt[k][v] = rank_level[m]
                            else:
                                vote_cnt[k][v] += rank_level[m]

                    if sum_result is None:
                        sum_result = temp
                    else:
                        sum_result = sum_result + temp

                result_predicts = sum_result
                final_result = result_predicts.argmax(1).cpu().numpy()

                for j, v_cnt in enumerate(vote_cnt):
                    bigest_cnt = 0
                    bigest_index = None
                    conflict_list = []
                    for k, v in v_cnt.items():
                        if bigest_index is None or bigest_cnt < v:
                            bigest_cnt = v
                            bigest_index = k
                            conflict_list = [k]
                        elif v == bigest_cnt:
                            conflict_list.append(k)

                    if len(conflict_list) == 1:
                        final_result[j]=bigest_index

            else:
                result_predicts = classifiers(features)
                final_result = result_predicts.argmax(1).cpu().numpy()


            allP.extend(final_result)
            allT.extend(labels.cpu().numpy())

    temp_all_t, temp_all_p = [], []
    for i in range(0, len(allT)):
        temp_all_t.append(int(allT[i]))
        temp_all_p.append(int(allP[i]))

    if evaluate == "micro" or evaluate == "macro":
        value = str(f1_score(allT, allP, average=evaluate))
    return value

def calc_result(config, is_show_all=False, choice_head_id=None, is_train=True, evaluate="micro"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Classifer(config).to(device)
    root_path = config["save_dir"]
    models_dir = os.path.join(root_path,"models_t1")
    global_state_dict = torch.load(os.path.join(models_dir, "global.pth"),weights_only=False)
    net.body.load_state_dict(global_state_dict, strict=True)
    net.eval()


    val_list = []
    data_path = config["dataset"]
    data_type = config["data_type"]
    with open(data_path, 'r') as file:
        data = json.load(file)
        for key, value in enumerate(data):
            if is_train:
                val_list += value["train_set"]
            else:
                val_list += value["test_set"]

    head_cnt = config["sdc_head_cnt"]
    if data_type == "ham":
        dataset = ham.init_with_obj(val_list)
    else:
        dataset = bds.init_with_obj(val_list, False)

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=8)

    for i in range(head_cnt):
        head = net.choice_heads[i]
        head_dict = torch.load(os.path.join(models_dir, f"personal_{i}.pth"),weights_only=False)
        head = head.to(device)
        head.load_state_dict(head_dict,strict=True)
        head.eval()
        net.choice_heads[i] = head


    if is_show_all:
        client_name = "spare heads ensemble"
        if choice_head_id is not None:
            final_choice_head = []
            for i, v in enumerate(net.choice_heads):
                if i in choice_head_id:
                    final_choice_head.append(v)
        else:
            final_choice_head = net.choice_heads[:head_cnt]

        t1 = calcResult(
            model=net,
            device=device,
            dataloader=dataloader,
            classifiers=final_choice_head,
            evaluate=evaluate,
        )
        print(f"\n{client_name} ====> {t1}\n")
        return t1
    else:
        all_acc = []
        for i, c in enumerate(net.choice_heads[:head_cnt]):
            client_name = f"client_{i}"
            t1 = calcResult(
                model=net,
                device=device,
                dataloader=dataloader,
                classifiers=c,
                evaluate=evaluate,
            )
            print(f"\n{client_name} ====> {t1}\n")
            all_acc.append((i,t1))

    return all_acc


if __name__ == "__main__":
    save_path = f"./result/training_result8/"
    config_path = os.path.join(save_path,"config.txt")
    with open(config_path, 'r') as file:
        config = json.load(file)

    gamma = 0.3
    head_cnt = int(config["sdc_head_cnt"])
    spare_cnt = math.ceil(gamma * head_cnt)
    target = None


    ensemble = calc_result(config, True, choice_head_id=None, is_train=False, evaluate="micro")

    all_model = calc_result(config, False, choice_head_id=None,is_train=True, evaluate="macro")
    sorted_data = sorted(all_model, key=lambda x: x[1], reverse=True)

    spare_heads = [sorted_data[i][0] for i in range(spare_cnt)]
    print("spare head ids", spare_heads)
    t2 = calc_result(config, True, choice_head_id=spare_heads,is_train=False,evaluate="micro")


