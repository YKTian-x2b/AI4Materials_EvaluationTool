from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import numpy as np

import paddle
from paddle.io import DataLoader, Dataset, Subset
from pymatgen.core.structure import Structure


def collate_pool(dataset_list):
    """
    Collate a list of dataConfig and return a batch
    Params:
        dataset_list: list of tuples for each dataConfig point.
        (atom_fea, nbr_fea, nbr_fea_idx, target)
    Returns:
        N = sum(n_i); N0 = sum(i)

        batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type
        batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        batch_nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        target: torch.Tensor shape (N, 1)
          Target value for prediction
        batch_cif_ids: list
    """
    batch_atom_in_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_in_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
        n_i = atom_in_fea.shape[0]  # number of atoms for this crystal
        batch_atom_in_fea.append(atom_in_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = paddle.to_tensor(np.arange(n_i, dtype=np.int64) + base_idx, place=paddle.CPUPlace(), dtype='int64')
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (paddle.concat(batch_atom_in_fea, axis=0),
            paddle.concat(batch_nbr_fea, axis=0),
            paddle.concat(batch_nbr_fea_idx, axis=0),
            crystal_atom_idx), \
            paddle.stack(batch_target, axis=0), batch_cif_ids


def get_train_val_test_loader(dataset, collate_fn,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, **kwargs):
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training dataConfig.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + valid_size]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    if return_test:
        test_indices = indices[train_size + valid_size:]
        test_dataset = Subset(dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        super().__init__()
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super().__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.
    Params:
        root_dir: str
            The path to the root directory of the dataset
        max_num_nbr: int
            The maximum number of neighbors while constructing the crystal graph
        radius: float
            The cutoff radius for searching neighbors
        dmin: float
            The minimum distance for constructing GaussianDistance
        step: float
            The step size for constructing GaussianDistance
        random_seed: int
            Random seed for shuffling the dataset
    Returns:
        atom_fea: (n_i, atom_fea_len)
        nbr_fea: (n_i, M, nbr_fea_len)
        nbr_fea_idx: (n_i, M)
        target: (1, )
        cif_id: str or int
    """

    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        super().__init__()
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        #
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        #
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))
        atom_in_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_in_fea = paddle.to_tensor(atom_in_fea, dtype='float32')
        nbr_fea_idx = paddle.to_tensor(nbr_fea_idx, dtype='int32')
        nbr_fea = paddle.to_tensor(nbr_fea, dtype='float32')
        target = paddle.to_tensor([float(target)], dtype='float32')
        return (atom_in_fea, nbr_fea, nbr_fea_idx), target, cif_id

    def __len__(self):
        return len(self.id_prop_data)

