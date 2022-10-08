import json
import datetime
import logging
from utils.trainer_keras import keras_train_model 
from utils.trainer_gpnas import GPNAS_train_model 
from utils.convert_x import convert_X_1 as convert_X

logging.basicConfig(level=logging.DEBUG, filename="test.log",
                    format=('%(filename)s: '
                            '%(levelname)s: '
                            '%(funcName)s(): '
                            '%(lineno)d:\t'
                            '%(message)s')
                    )

rank_name_list = ['cplfw_rank', 'market1501_rank', 'dukemtmc_rank',
                  'msmt17_rank', 'veri_rank', 'vehicleid_rank', 'veriwild_rank', 'sop_rank']

# rank_name_list = ['cplfw_rank']

def data_loader(fpath):
    '''
    load data from json
    return:
    '''
    with open(fpath, 'r') as f:
        input_data = json.load(f)
    rank_list = {name: [] for name in rank_name_list}
    arch_list = []

    keys_list = input_data.keys()
    for key in input_data.keys():
        for idx, name in enumerate(rank_name_list):
            rank_list[name].append(input_data[key][name])
        arch_list.append(convert_X(input_data[key]['arch']))
    return input_data, arch_list, rank_list


def data_output(data_ori, predict_rank, suff = ""):
    for idx, key in enumerate(data_ori.keys()):
        for name in rank_name_list:
            if name in predict_rank:
                data_ori[key][name] = int(predict_rank[name][idx])
    suff = datetime.datetime.now().strftime("%m%d_%H%M") + suff
    path = f'./data/CCF_UFO_submit_A_{suff}.json'
    logging.info(f"write result into {path}")
    with open(path, 'w') as f:
        json.dump(data_ori, f, indent=4)


def tarin_ex(name, ex_name):
    print(f"tarin_ex -------- {name} --------  ex:{ex_name }")
    model = GPNAS_train_model()

    nw_input = []
    for i in range(len(data_input_X)):
        X = data_input_X[i]
        nw_input.append([data_input_Y[ex_name][i]] + X)
    acc = model.do_train(data_input_X, data_input_Y[name])

    if models[name].model_acc < model.model_acc:
        models[name] = model

def with_GPNAS():
    print(f"======== GPNAS start training ========")
    for idx, name in enumerate(rank_name_list):
        print(f"-------- {name} --------")
        model = GPNAS_train_model()
        acc = model.do_train(data_input_X, data_input_Y[name])
        models[name] = model

    print(f"====== GPNAS predict acc ======")
    for idx, name in enumerate(rank_name_list):
        print(f"    {name} acc:{models[name].model_acc}")
    print(" ")


def with_keras(do_predict = False):
    print(f"======== keras start training ========")
    predict_rank = {name: [0 for i in range(len(data_test_X))] for name in rank_name_list}

    done_str = ""
    for idx, name in enumerate(rank_name_list):
        logging.info(f"-------- now doing: {name} --------")
        print(f"-------- {name} --------")
        model = keras_train_model()
        acc = model.do_train(data_input_X, data_input_Y[name])
        models[name] = model

        if do_predict:
            rank = model.do_predict(data_test_X)
            predict_rank[name] = rank
            done_str = done_str + str(name) + '-'
            data_output(data_test_ori, predict_rank, f"_{done_str[:-1]}")

    print(f"====== keras predict acc ======")
    for idx, name in enumerate(rank_name_list):
        print(f"    {name} acc:{models[name].model_acc}")
    print(" ")


def main():
    global data_input_X, data_input_Y
    data_input_ori, data_input_X, data_input_Y = data_loader(
        'data/data162979/CCF_UFO_train.json')

    global data_test_ori, data_test_X
    data_test_ori, data_test_X, data_test_Y = data_loader(
        'data/data162979/CCF_UFO_test.json')

    global models
    models = {}

    # with_GPNAS()
    with_keras(do_predict=True)

    print(f"====== predict acc ======")
    for idx, name in enumerate(rank_name_list):
        print(f"    {name} acc:{models[name].model_acc}")
    print(" ")


    return
    print(f"======== start predicting ========")
    predict_rank = {name: [] for name in rank_name_list}

    for idx, name in enumerate(rank_name_list):
        print(f"-------- {name} --------")
        rank = models[name].do_predict(data_test_X)
        predict_rank[name] = rank

    data_output(data_test_ori, predict_rank)


if __name__ == "__main__":
    main()
