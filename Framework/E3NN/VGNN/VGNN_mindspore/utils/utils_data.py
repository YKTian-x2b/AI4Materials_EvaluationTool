import mindspore as ms
import mindspore.dataset as ds
import glob
import os
import pandas as pd
import numpy as np
import pickle as pkl
from ase.neighborlist import neighbor_list
from ase import Atom
import mendeleev as md
import itertools
from copy import copy

default_dtype = ms.float64


def build_data_vvn(id, structure, qpts, gphonon, r_max, vnelem='Fe', descriptor='mass'):
    symbols = list(structure.symbols).copy()
    positions = ms.Tensor.from_numpy(structure.get_positions().copy())
    numb = len(positions)
    lattice = ms.Tensor.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    _edge_src, _edge_dst, _edge_shift, _, _ = neighbor_list("ijSDd", a=structure, cutoff=r_max, self_interaction=True)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len, structure_vn = create_virtual_nodes_vvn(structure, vnelem,
                                                                                                _edge_src, _edge_dst,
                                                                                                _edge_shift)
    z = get_node_attr_vvn(structure_vn.arrays['numbers'])
    x = get_node_feature_vvn(structure_vn.arrays['numbers'], descriptor)
    node_deg = get_node_deg(edge_dst, len(x))
    y = ms.Tensor.from_numpy(gphonon / 1000).unsqueeze(0)
    data = Data(id=id,
                pos=positions,
                lattice=lattice,
                symbol=symbols,
                x=x,
                z=z,
                y=y,
                node_deg=node_deg,
                edge_index=ms.ops.stack([ms.tensor(edge_src, ms.int64), ms.tensor(edge_dst, ms.int64)], dim=0),
                edge_shift=ms.tensor(edge_shift, dtype=default_dtype),
                edge_vec=ms.tensor(edge_vec, dtype=default_dtype),
                edge_len=ms.tensor(edge_len, dtype=default_dtype),
                qpts=ms.from_numpy(qpts).unsqueeze(0),
                gphonon=ms.from_numpy(gphonon).unsqueeze(0),
                r_max=r_max,
                # ucs = None,
                numb=numb)

    return data


def generate_gamma_data_dict(data_dir, run_name, data, r_max, vn_an=26, descriptor='mass'):
    data_dict_path = os.path.join(data_dir, f'data_dict_{run_name}.pkl')
    vnelem = Atom(vn_an).symbol #!
    if len(glob.glob(data_dict_path)) == 0:
        data_dict = dict()
        ids = data['id']
        structures = data['structure']
        qptss = data['qpts']
        band_structures = data['band_structure']
        for id, structure, qpts, band_structure in zip(ids, structures, qptss, band_structures):
            # print(id)
            gi = np.argmin(np.abs(np.linalg.norm(qpts - np.array([0, 0, 0]), axis = 1)), axis = 0)
            data_dict[id] = build_data_vvn(id, structure, qpts[gi], band_structure[gi], r_max, vnelem, descriptor)
        # pkl.dump(data_dict, open(data_dict_path, 'wb'))
    else:
        data_dict  = pkl.load(open(data_dict_path, 'rb'))
    return data_dict