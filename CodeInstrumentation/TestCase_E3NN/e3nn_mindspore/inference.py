import mindspore as ms
import mindspore.dataset as ds
from utils import load_band_structure_data, generate_gamma_data_dict_ms, msDataset
from model import GraphNetworkVVN, BandLoss
import time
import os


def evaluate(model, dataloader, len_dataloader, loss_fn):
    model.set_train(False)
    loss_cumulative = 0.
    for i, d in enumerate(dataloader):
        logits = model(d)
        loss = loss_fn(logits, d["y"])
        loss_cumulative += loss.asnumpy()   # [0]
    return loss_cumulative / len_dataloader


def inference():
    ms.set_context(mode=ms.PYNATIVE_MODE,
                   save_graphs=False, save_graphs_path="./graphs",
                   device_target="GPU", device_id=0)
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

    run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
    data_dir = '../data'
    raw_dir = '../data/phonon'
    data_file = 'DFPT_band_structure.pkl'

    data = load_band_structure_data(data_dir, raw_dir, data_file)
    data_dict = generate_gamma_data_dict_ms(data_dir, run_name, data, r_max, vn_an)
    num = len(data_dict)
    dataset_row = list(data_dict.values())
    data_set_generator = msDataset(dataset_row)
    dataset = ds.GeneratorDataset(data_set_generator, ["id",
                                                       "pos",
                                                       "lattice",
                                                       "symbol",
                                                       "x",
                                                       "z",
                                                       "y",
                                                       "node_deg",
                                                       "edge_index",
                                                       "edge_shift",
                                                       "edge_vec",
                                                       "edge_len",
                                                       "qpts",
                                                       "gphonon",
                                                       "r_max",
                                                       "numb", ], shuffle=True)
    dataset.batch(batch_size=batch_size)
    dataset_loader = dataset.create_dict_iterator()
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
    test_avg_loss = evaluate(model, dataset_loader, num, loss_fn)
    print("test_avg_loss: ", test_avg_loss)


if __name__ == '__main__':
    # 调整工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.chdir(current_dir)
    inference()
