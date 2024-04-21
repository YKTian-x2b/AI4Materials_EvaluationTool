import paddle
from paddle import nn

if __name__ == "__main__":
    N = 2
    M = 3
    hidden_fea_len = 4
    # atoms_in_fea: (N, atom_fea_len）
    # atom_bonds_in_fea: (N, M, bond_fea_len)
    # atom_nbrs_idx: (N, M)

    atoms_in_fea = paddle.to_tensor([[0.1, 0.3, 0.2, 0.4], [0.5, 0.6, 0.8, 0.7]])
    atom_nbrs_idx = paddle.to_tensor([[0, 1, 0], [1, 1, 0]])
    atom_nbr_fea = atoms_in_fea[atom_nbrs_idx, :]
    print(atom_nbr_fea.shape)