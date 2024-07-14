import torch
import time
import os
from utils_load import load_band_structure_data
from utils_data import generate_gamma_data_dict
from model import BandLoss, GraphNetworkVVN, evaluate
from torch_geometric.loader import DataLoader


def inference():
    torch.set_default_dtype(torch.float32)
    device = 'cuda'

    run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
    data_dir = '../data'
    raw_dir = '../data/phonon'
    data_file = 'DFPT_band_structure.pkl'

    batch_size = 1
    lmax = 0
    mul = 4
    nlayers = 1
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

    data = load_band_structure_data(data_dir, raw_dir, data_file)
    data_dict = generate_gamma_data_dict(data_dir, run_name, data, r_max, vn_an)
    num = len(data_dict)
    data_set = torch.utils.data.Subset(list(data_dict.values()), range(num))

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
    loss_fn = BandLoss()
    dataloader = DataLoader(data_set, batch_size=batch_size)
    test_avg_loss = evaluate(model, dataloader, loss_fn, device)
    print("test_avg_loss: ", test_avg_loss)


if __name__ == '__main__':
    # 调整工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.chdir(current_dir)
    inference()