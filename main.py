import json
import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.trainer import GPNAS_train_model
from utils.convert_x import convert_X_1 as convert_X

rank_name_list = ['cplfw_rank', 'market1501_rank', 'dukemtmc_rank',
                  'msmt17_rank', 'veri_rank', 'vehicleid_rank', 'veriwild_rank', 'sop_rank']


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


def data_output(data_ori, predict_rank):
    # predict_data = {}
    for idx,key in enumerate(data_ori.keys()):
        for name in rank_name_list:
            data_ori[key][name] = int(1000*predict_rank[name][idx])
            # predict_data.update(
            #     {key: {name: int(1000*predict_rank[name][idx])}})
    suff = datetime.datetime.now().strftime("%m%d_%H%M")
    with open(f'./CCF_UFO_submit_A_{suff}.json', 'w') as f:
        json.dump(data_ori, f, indent=4)


def main():
    data_input_ori, data_input_X, data_input_Y = data_loader(
        'data/data162979/CCF_UFO_train.json')

    models = {}
    models_acc = {}
    
    print(f"======== start training ========")
    for idx, name in enumerate(rank_name_list):
        print(f"-------- {name} --------")
        model = GPNAS_train_model()
        acc = model.do_train(data_input_X, data_input_Y[name])
        models[name] = model

    # def add_model(name):
    #     model = GPNAS_train_model()
    #     model.name = name
    #     acc = model.do_train(data_input_X.copy(), data_input_Y[name].copy())
    #     models[name] = model
    #     models_acc[name] = acc
    # th_list = []
    # for idx, name in enumerate(rank_name_list):
    #     th = threading.Thread(name=name, target= add_model, kwargs={"name":name})
    #     th.start()
    #     th_list.append(th)

    # for th in th_list:
    #     th.join()

    # with ThreadPoolExecutor(12) as executor:
    #     th_list = []
    #     def add_model(name):
    #         model = GPNAS_train_model()
    #         model.name = name
    #         acc = model.do_train(data_input_X.copy(), data_input_Y[name].copy())
    #         return name, acc, model

    #     for idx, name in enumerate(rank_name_list):
    #         th = executor.submit(add_model, name)
    #         th_list.append(th)

    #     for th in th_list:
    #         name, acc, model = th.result()
    #         print(f"-------- {name} --------")
    #         models[name] = model
    #         models_acc[name] = acc
            

    data_test_ori, data_test_X, data_test_Y = data_loader(
        'data/data162979/CCF_UFO_test.json')

    print(f"======== start predicting ========")
    predict_rank = {name: [] for name in rank_name_list}

    
    for idx, name in enumerate(rank_name_list):
        print(f"-------- {name} --------")
        rank = models[name].do_predict(data_test_X)
        predict_rank[name] = rank

    # with ThreadPoolExecutor(12) as executor:
    #     th_list = []
    #     def predict(name):
    #         rank = models[name].do_predict(data_test_X)
    #         return name, rank

    #     for idx, name in enumerate(rank_name_list):
    #         th = executor.submit(predict, name)
    #         th_list.append(th)

    #     for th in th_list:
    #         name, rank = th.result()
    #         print(f"-------- {name} --------")
    #         predict_rank[name] = rank

    data_output(data_test_ori, predict_rank)


if __name__ == "__main__":
    main()
