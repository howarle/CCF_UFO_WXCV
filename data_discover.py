import json
import random
import numpy as np
import scipy.stats
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

def test():
    data_input_ori, data_input_X, data_input_Y = data_loader(
        'data/data162979/CCF_UFO_train.json')
    
    data = {}
    for name in rank_name_list:
        tmp = [tuple((rk, i)) for i, rk in enumerate(data_input_Y[name])]
        tmp.sort()
        data[name] = [data_input_X[i] for rk, i in tmp]

    
    out = json.dumps(data, separators=(',', ':'), indent=4)
    out = out.replace(',\n            ', ',')
    out = out.replace('\n        ],', '],')
    out = out.replace('[\n            ', '[')
    open("./data_discover.json", "w").write(out)

    rank_sa = []
    for name1 in rank_name_list:
        rank_sa1 = []
        for name2 in rank_name_list:
            acc = scipy.stats.kendalltau(data[name1], data[name2]).correlation
            rank_sa1.append(acc)
        rank_sa.append(rank_sa1)
    rank_sa = np.array(rank_sa)

    print(rank_sa)

if __name__ == "__main__":
    test()
