import torch
import torch_geometric
import torch_scatter

import e3nn
import math
# from e3nn import rs, o3
from e3nn.point.data_helpers import DataPeriodicNeighbors
from e3nn.networks import GatedConvParityNetwork
from e3nn.kernel_mod import Kernel
from e3nn.point.message_passing import Convolution

import pymatgen
from pymatgen.core.structure import Structure
import numpy as np
import time, os
import datetime
import pickle
from pymatgen.core.periodic_table import Element
import matplotlib.pyplot as plt


class AtomEmbeddingAndSumLastLayer(torch.nn.Module):
    def __init__(self, atom_type_in, atom_type_out, model):
        super().__init__()
        self.linear = torch.nn.Linear(atom_type_in, atom_type_out)
        self.model = model
        self.relu = torch.nn.ReLU()

    def forward(self, x, *args, batch=None, **kwargs):
        output = self.linear(x)
        output = self.relu(output)
        output = self.model(output, *args, **kwargs)
        if batch is None:
            N = output.shape[0]
            batch = output.new_ones(N)
        output = torch_scatter.scatter_add(output, batch, dim=0)
        output = self.relu(output)
        maxima, _ = torch.max(output, axis=1)
        output = output.div(maxima.unsqueeze(1))
        return output


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate(model, dataloader, device):
    model.eval()
    loss_cumulative = 0.
    loss_cumulative_mae = 0.
    start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            loss = loss_fn(output, d.y).cpu()
            loss_mae = loss_fn_mae(output, d.y).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    return loss_cumulative / len(dataloader), loss_cumulative_mae / len(dataloader)


def train(model, optimizer, dataloader, dataloader_valid, max_iter=101., device="cpu"):
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    dynamics = []

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mae = 0.
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            loss = loss_fn(output, d.y).cpu()
            loss_mae = loss_fn_mae(output, d.y).cpu()
            print(f"Iteration {step + 1:4d}    batch {j + 1:5d} / {len(dataloader):5d}   " +
                  f"batch loss = {loss.data}", end="\r", flush=True)
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        end_time = time.time()
        wall = end_time - start_time

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, dataloader_valid, device)
            # This is the more correct thing to do
            # -- but since evaluation takes long, we will skip it and use during batch values.
            train_avg_loss = evaluate(model, dataloader, device)

            dynamics.append({
                'step': step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                    'mean_abs': loss_mae.item(),
                },
                'valid': {
                    'loss': valid_avg_loss[0],
                    'mean_abs': valid_avg_loss[1],
                },
                #                 'train': {
                #                     'loss': loss_cumulative / len(dataloader),
                #                     'mean_abs': loss_cumulative_mae / len(dataloader),
                #                 },
                'train': {
                    'loss': train_avg_loss[0],
                    'mean_abs': train_avg_loss[1],
                },
            })

            yield {
                'dynamics': dynamics,
                'state': model.state_dict()
            }

            print(f"Iteration {step + 1:4d}    batch {j + 1:5d} / {len(dataloader):5d}   " +
                  f"train loss = {train_avg_loss[0]:8.3f}   " +
                  f"valid loss = {valid_avg_loss[0]:8.3f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")
        scheduler.step()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('torch device:', device)

    params = {'len_embed_feat': 64,
              'num_channel_irrep': 32,
              'num_e3nn_layer': 2,
              'max_radius': 5,
              'num_basis': 10,
              'adamw_lr': 0.005,
              'adamw_wd': 0.05
              }
    params_name = '_len51max1000_fwin101ord3'

    par_directory = 'models/' + time.strftime("%y%m%d", time.localtime())
    if not os.path.isdir(par_directory):
        os.mkdir(par_directory)
    run_name = ('models/' + time.strftime("%y%m%d", time.localtime()) + '/'
                + time.strftime("%y%m%d-%H%M", time.localtime()) + params_name)

    with open('models/phdos_e3nn_len51max1000_fwin101ord3.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    print(data_dict.keys())

    material_id = data_dict['material_id']
    cif = data_dict['cif']
    phfre = data_dict['phfre']
    phdos = data_dict['phdos']
    phfre_gt = data_dict['phfre_gt']
    phdos_gt = data_dict['phdos_gt']
    print(len(cif))

    phdos = torch.tensor(phdos)
    structures = [Structure.from_str("\n".join(c), "CIF") for c in cif]

    species = set()
    for struct in structures[:]:
        species = species.union(list(set(map(str, struct.species))))
    species = sorted(list(species))

    len_element = 118
    atom_types_dim = len_element
    embedding_dim = params['len_embed_feat']
    len_dos = phdos.shape[1]
    lmax = 1
    n_norm = 40  # Roughly the average number (over entire dataset) of nearest neighbors for a given atom

    Rs_in = [(embedding_dim, 0, 1)]  # num_atom_types scalars (L=0) with even parity
    Rs_out = [(len_dos, 0, 1)]  # len_dos scalars (L=0) with even parity

    model_kwargs = {
        "convolution": Convolution,
        "kernel": Kernel,
        "Rs_in": Rs_in,
        "Rs_out": Rs_out,
        "mul": params['num_channel_irrep'],  # number of channels per irrep (differeing L and parity)
        "layers": params['num_e3nn_layer'],
        "max_radius": params['max_radius'],
        "lmax": lmax,
        "number_of_basis": params['num_basis']
    }
    print(model_kwargs)

    model = AtomEmbeddingAndSumLastLayer(atom_types_dim, embedding_dim, GatedConvParityNetwork(**model_kwargs))
    opt = torch.optim.AdamW(model.parameters(), lr=params['adamw_lr'], weight_decay=params['adamw_wd'])

    data = []
    for i, struct in enumerate(structures):
        print(f"Encoding sample {i + 1:5d}/{len(structures):5d}", end="\r", flush=True)
        input = torch.zeros(len(struct), len_element)
        for j, site in enumerate(struct):
            input[j, int(Element(str(site.specie)).Z)] = Element(str(site.specie)).atomic_mass
        data.append(DataPeriodicNeighbors(
            x=input, Rs_in=None,
            pos=torch.tensor(struct.cart_coords.copy()), lattice=torch.tensor(struct.lattice.matrix.copy()),
            r_max=params['max_radius'], y=phdos[i].unsqueeze(0), n_norm=n_norm,
        ))

    torch.save(data, run_name + '_data.pt')

    with open('models/200801_trteva_indices.pkl', 'rb') as f:
        index_tr, index_va, index_te = pickle.load(f)

    assert set(index_tr).isdisjoint(set(index_te))
    assert set(index_tr).isdisjoint(set(index_va))
    assert set(index_te).isdisjoint(set(index_va))

    batch_size = 1
    dataloader = torch_geometric.data.DataLoader([data[i] for i in index_tr], batch_size=batch_size, shuffle=True)
    dataloader_valid = torch_geometric.data.DataLoader([data[i] for i in index_va], batch_size=batch_size)

    loss_fn = torch.nn.MSELoss()
    loss_fn_mae = torch.nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

    for results in train(model, opt, dataloader, dataloader_valid, device=device, max_iter=64):
        with open(run_name + '_trial_run_full_data.torch', 'wb') as f:
            results['model_kwargs'] = model_kwargs
            torch.save(results, f)

    saved = torch.load(run_name + '_trial_run_full_data.torch')
    steps = [d['step'] + 1 for d in saved['dynamics']]
    valid = [d['valid']['loss'] for d in saved['dynamics']]
    train = [d['train']['loss'] for d in saved['dynamics']]
    plt.plot(steps, train, 'o-', label="train")
    plt.plot(steps, valid, 'o-', label="valid")
    # plt.yscale("log")
    plt.legend()
    plt.savefig(run_name + '_hist.png', dpi=300)
