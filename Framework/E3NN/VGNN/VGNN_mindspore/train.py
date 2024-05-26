import mindspore as ms
from mindspore.experimental import optim
import mindspore.dataset as ds
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from model import train, GraphNetworkVVN, BandLoss
from utils.utils_load import load_band_structure_data
from utils.utils_data import generate_gamma_data_dict
import random


def subset_generator(original_dataset, indices):
    resList = []
    for idx in indices:
        resList.append(original_dataset[idx])
    return resList


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE,
                   save_graphs=False, save_graphs_path="./graphs",
                   device_target="GPU", device_id=0)
    tr_ratio = 0.9
    batch_size = 1
    k_fold = 5

    max_iter = 100  # 200
    lmax = 0
    mul = 4
    nlayers = 4
    r_max = 7
    number_of_basis = 5
    radial_layers = 1
    radial_neurons = 65
    node_dim = 118
    node_embed_dim = 8
    input_dim = 118
    input_embed_dim = 18
    vn_an = 26
    irreps_out = '1x0e'
    option = 'vvn'

    loss_fn = BandLoss()
    lr = 0.005
    weight_decay = 0.05
    schedule_gamma = 0.96

    run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
    model_dir = 'models'
    data_dir = 'data'
    raw_dir = './data/phonon'
    data_file = 'DFPT_band_structure.pkl'

    os.system(f'rm -r {data_dir}/9850858*')
    os.system(f'rm -r {data_dir}/phonon/')
    os.system(f'cd {data_dir}; wget --no-verbose https://figshare.com/ndownloader/files/9850858')
    os.system(f'cd {data_dir}; tar -xf 9850858')
    os.system(f'rm -r {data_dir}/9850858*')

    data = load_band_structure_data(data_dir, raw_dir, data_file)
    data_dict = generate_gamma_data_dict(data_dir, run_name, data, r_max, vn_an)

    num = len(data_dict)
    tr_nums = [int((num * tr_ratio) // k_fold)] * k_fold
    te_num = num - sum(tr_nums)
    idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)
    with open(f'./data/idx_{run_name}_tr.txt', 'w') as f:
        for idx in idx_tr: f.write(f"{idx}\n")
    with open(f'./data/idx_{run_name}_te.txt', 'w') as f:
        for idx in idx_te: f.write(f"{idx}\n")

    # data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
    # tr_set, te_set = torch.utils.data.Subset(data_set, idx_tr), torch.utils.data.Subset(data_set, idx_te)
    data_set = subset_generator(list(data_dict.values()), range(len(data_dict)))
    tr_set = subset_generator(data_set, idx_tr)
    te_set = subset_generator(data_set, idx_te)

    model = GraphNetworkVVN(mul,
                            irreps_out,
                            lmax,
                            nlayers,
                            number_of_basis,
                            radial_layers,
                            radial_neurons,
                            node_dim,
                            node_embed_dim,
                            input_dim,
                            input_embed_dim)

    opt = optim.AdamW(model.trainable_params(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=schedule_gamma)
    train(model,
          opt,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          batch_size,
          k_fold,
          option=option)


if __name__ == '__main__':
    main()
