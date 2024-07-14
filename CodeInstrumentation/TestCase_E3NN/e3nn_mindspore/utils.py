import json
from ase import Atoms
from pymatgen.core.structure import Structure
import mindspore as ms
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

default_dtype = ms.float32


def load_band_structure_data(data_dir, raw_dir, data_file):
    data_path = os.path.join(data_dir, data_file)
    if len(glob.glob(data_path)) == 0:
        df = pd.DataFrame({})
        for file_path in glob.glob(os.path.join(raw_dir, '*.json')):
            Data = dict()
            with open(file_path) as f:
                data = json.load(f)
            structure = Structure.from_str(data['metadata']['structure'], fmt = 'cif')
            # print(structure.__dict__)
            atoms = Atoms(list(map(lambda x: x.symbol, structure.species)),
                            positions = structure.cart_coords.copy(),
                            cell = structure.lattice.matrix.copy(),
                            pbc=True)
            Data['id'] = data['metadata']['material_id']
            Data['structure'] = [atoms]
            Data['qpts'] = [np.array(data['phonon']['qpts'])]
            Data['band_structure'] = [np.array(data['phonon']['ph_bandstructure'])]
            dfn = pd.DataFrame(data = Data)
            df = pd.concat([df, dfn], ignore_index = True)
        df.to_pickle(data_path)
        return df
    else:
        return pd.read_pickle(data_path)




def append_diag_vn(struct, element="Fe"):
    # diagonal virtual nodes
    # option for atom choices.
    """_summary_
    Args:
        struct (ase.atoms.Atoms): original ase.atoms.Atoms object
    Returns:
        ase.atoms.Atoms: Atoms affter appending additonal nodes
    """
    cell = struct.get_cell()
    num_sites = struct.get_positions().shape[0]
    total_len = 3 * num_sites
    struct2 = struct.copy()
    for i in range(total_len):
        vec = i * (cell[0] + cell[1] + cell[2]) / total_len
        struct2.append(Atom(element, (vec[0], vec[1], vec[2])))
    return struct2


def atom_feature(atomic_number: int, descriptor):
    """_summary_

    Args:
        atomic_number (_int_): atomic number
        descriptor (_'str'_): descriptor type. select from ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']

    Returns:
        _type_: descriptor
    """
    if descriptor == 'mass':  # Atomic Mass (amu)
        feature = Atom(atomic_number).mass
    elif descriptor == 'number':  # atomic number
        feature = atomic_number
    else:
        ele = md.element(atomic_number)  # use mendeleev
        if descriptor == 'radius':  # Atomic Radius (pm)
            feature = ele.atomic_radius
        elif descriptor == 'en':  # Electronegativity (Pauling)
            feature = ele.en_pauling
        elif descriptor == 'ie':  # Ionization Energy (eV)
            feature = ele.ionenergies[1]
        elif descriptor == 'dp':  # Dipole Polarizability (Å^3)
            feature = ele.dipole_polarizability
        else:  # no feature
            feature = 1
    return feature


def create_virtual_nodes_vvn(structure0, vnelem, edge_src0, edge_dst0, edge_shift0):
    structure = append_diag_vn(structure0, element=vnelem)
    positions = ms.Tensor.from_numpy(structure.get_positions().copy())
    positions0 = ms.Tensor.from_numpy(structure0.get_positions().copy())
    numb = len(positions0)
    lattice = ms.Tensor.from_numpy(structure.cell.array.copy()).unsqueeze(0).astype(default_dtype)
    idx_real, idx_virt = range(numb), range(numb, 4 * numb)
    rv_pairs = list(itertools.product(idx_real, idx_virt))
    vv_pairs = list(itertools.product(idx_virt, idx_virt))
    edge_src = copy(edge_src0)
    edge_dst = copy(edge_dst0)
    edge_shift = copy(edge_shift0)
    for i in range(len(rv_pairs)):
        edge_src = np.append(edge_src, np.array([rv_pairs[i][0]]))
        edge_dst = np.append(edge_dst, np.array([rv_pairs[i][1]]))
        edge_shift = np.concatenate((edge_shift, np.array([[0, 0, 0]])), axis=0)
    for j in range(len(vv_pairs)):
        edge_src = np.append(edge_src, np.array([vv_pairs[j][0]]))
        edge_dst = np.append(edge_dst, np.array([vv_pairs[j][1]]))
        edge_shift = np.concatenate((edge_shift, np.array([[0, 0, 0]])), axis=0)
    edge_batch = positions.new_zeros(positions.shape[0], dtype=ms.int32)[ms.Tensor.from_numpy(edge_src)]
    edge_vec = (positions[ms.Tensor.from_numpy(edge_dst)]
                - positions[ms.Tensor.from_numpy(edge_src)]
                + ms.ops.einsum('ni,nij->nj', ms.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    return edge_src, edge_dst, edge_shift, edge_vec, edge_len, structure


def get_node_attr_vvn(atomic_numbers):
    z = []
    for atomic_number in atomic_numbers:
        node_attr = [0.0] * 118
        node_attr[atomic_number - 1] = 1
        z.append(node_attr)
    return ms.Tensor.from_numpy(np.array(z, dtype=np.float32))


def get_node_feature_vvn(atomic_numbers, descriptor='mass'):
    x = []
    for atomic_number in atomic_numbers:
        node_feature = [0.0] * 118
        node_feature[atomic_number - 1] = atom_feature(int(atomic_number), descriptor)
        x.append(node_feature)
    return ms.Tensor.from_numpy(np.array(x, dtype=np.float32))


def get_node_deg(edge_dst, n):
    node_deg = np.zeros((n, 1), dtype=np.float32)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += node_deg == 0
    return ms.Tensor.from_numpy(node_deg)


class msData():
    def __init__(self,
                 id,
                 pos: ms.Tensor,
                 lattice: ms.Tensor,
                 symbol: list,
                 x: ms.Tensor,
                 z: ms.Tensor,
                 y: ms.Tensor,
                 node_deg: ms.Tensor,
                 edge_index: ms.Tensor,
                 edge_shift: ms.Tensor,
                 edge_vec: ms.Tensor,
                 edge_len: ms.Tensor,
                 qpts: ms.Tensor,
                 gphonon: ms.Tensor,
                 r_max,
                 numb):
        setattr(self, "id", id)
        setattr(self, "pos", pos)
        setattr(self, "lattice", lattice)
        setattr(self, "symbol", symbol)
        setattr(self, "x", x)
        setattr(self, "z", z)
        setattr(self, "y", y)
        setattr(self, "node_deg", node_deg)
        setattr(self, "edge_index", edge_index)
        setattr(self, "edge_shift", edge_shift)
        setattr(self, "edge_vec", edge_vec)
        setattr(self, "edge_len", edge_len)
        setattr(self, "qpts", qpts)
        setattr(self, "gphonon", gphonon)
        setattr(self, "r_max", r_max)
        setattr(self, "numb", numb)

    def __getitem__(self, item: str):
        return getattr(self, item)


def build_data_vvn_ms(id, structure, qpts, gphonon, r_max, vnelem='Fe', descriptor='mass'):
    symbols = list(structure.symbols).copy()
    positions = ms.Tensor.from_numpy(structure.get_positions().copy()).astype(default_dtype)
    numb = len(positions)
    lattice = ms.Tensor.from_numpy(structure.cell.array.copy()).unsqueeze(0).astype(default_dtype)
    _edge_src, _edge_dst, _edge_shift, _, _ = neighbor_list("ijSDd", a=structure, cutoff=r_max, self_interaction=True)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len, structure_vn = create_virtual_nodes_vvn(structure, vnelem,
                                                                                                _edge_src, _edge_dst,
                                                                                                _edge_shift)
    z = get_node_attr_vvn(structure_vn.arrays['numbers'])
    x = get_node_feature_vvn(structure_vn.arrays['numbers'], descriptor)
    node_deg = get_node_deg(edge_dst, len(x))
    y = ms.Tensor.from_numpy(gphonon / 1000).unsqueeze(0).astype(default_dtype)
    data = msData(id=id,
                  pos=positions,
                  lattice=lattice,
                  symbol=symbols,
                  x=x,
                  z=z,
                  y=y,
                  node_deg=node_deg,
                  edge_index=ms.ops.stack([ms.tensor(edge_src, ms.int32), ms.tensor(edge_dst, ms.int32)], axis=0),
                  edge_shift=ms.tensor(edge_shift, dtype=default_dtype),
                  edge_vec=ms.tensor(edge_vec, dtype=default_dtype),
                  edge_len=ms.tensor(edge_len, dtype=default_dtype),
                  qpts=ms.Tensor.from_numpy(qpts).unsqueeze(0).astype(default_dtype),
                  gphonon=ms.Tensor.from_numpy(gphonon).unsqueeze(0).astype(default_dtype),
                  r_max=r_max,
                  numb=numb)
    return data


def generate_gamma_data_dict_ms(data_dir, run_name, data, r_max, vn_an=26, descriptor='mass'):
    data_dict_path = os.path.join(data_dir, f'data_dict.pkl')
    vnelem = Atom(vn_an).symbol  # !
    if len(glob.glob(data_dict_path)) == 0:
        data_dict = dict()
        ids = data['id']
        structures = data['structure']
        qptss = data['qpts']
        band_structures = data['band_structure']
        for id, structure, qpts, band_structure in zip(ids, structures, qptss, band_structures):
            # print(id)
            gi = np.argmin(np.abs(np.linalg.norm(qpts - np.array([0, 0, 0]), axis=1)), axis=0)
            data_dict[id] = build_data_vvn_ms(id, structure, qpts[gi], band_structure[gi], r_max, vnelem, descriptor)
        pkl.dump(data_dict, open(data_dict_path, 'wb'))
    else:
        print("pkl.load")
        data_dict = pkl.load(open(data_dict_path, 'rb'))
    return data_dict


class msDataset():
    def __init__(self, rawDataset):
        self.data = rawDataset

    def __getitem__(self, index):
        return self.data[index]["id"], \
            self.data[index]["pos"], \
            self.data[index]["lattice"], \
            self.data[index]["symbol"], \
            self.data[index]["x"], \
            self.data[index]["z"], \
            self.data[index]["y"], \
            self.data[index]["node_deg"], \
            self.data[index]["edge_index"], \
            self.data[index]["edge_shift"], \
            self.data[index]["edge_vec"], \
            self.data[index]["edge_len"], \
            self.data[index]["qpts"], \
            self.data[index]["gphonon"], \
            self.data[index]["r_max"], \
            self.data[index]["numb"]

    def __len__(self):
        return len(self.data)