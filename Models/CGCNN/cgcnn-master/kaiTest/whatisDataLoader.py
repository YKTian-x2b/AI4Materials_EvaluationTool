from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np


def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
        torch.stack(batch_target, dim=0), \
        batch_cif_ids


def learn_dataloader(dataset, collate_pool):
    total_size = len(dataset)
    train_size = 0.6 * total_size
    batch_size = 256
    num_workers = 8
    pin_memory = True
    # sampler是采样索引用的
    indices = list(range(total_size))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    # collate_fn是自定义函数，根据索引getitem，然后堆叠成batch
    # 如果abc是属性，数据集是[(a,b,c), (a,b,c) ...]的话，batch就是 ([a,a,a...],[b,b,b...],[c,c,c...])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_pool, pin_memory=pin_memory)


if __name__ == '__main__':
    dataset = []
    (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id = dataset[0]
    learn_dataloader(dataset, collate_pool)