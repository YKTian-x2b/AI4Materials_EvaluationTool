import paddle
from paddle import nn
import nvtx


class ConvLayer(nn.Layer):
    def __init__(self, atom_fea_len_, bond_fea_len_):
        """
        atom_fea_len: 原子特征长度
        bond_fea_len: 键特征长度
        """
        super().__init__()
        self.atom_fea_len = atom_fea_len_
        self.bond_fea_len = bond_fea_len_
        self.fc = nn.Linear(2 * atom_fea_len_ + bond_fea_len_,
                            2 * atom_fea_len_)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1D(2 * atom_fea_len_)
        self.bn2 = nn.BatchNorm1D(atom_fea_len_)
        self.softplus2 = nn.Softplus()

    def forward(self, atoms_in_fea_, atom_bonds_in_fea_, atom_nbrs_idx_):
        """
        N: total num of atoms in the batch
        M: Max num of neighbors
        atoms_in_fea: (N, atom_fea_len）
        atom_bonds_in_fea: (N, M, bond_fea_len)
        atom_nbrs_idx: (N, M)
        """
        # CGCNN式5 非常清晰的描述了线性卷积的实现过程
        N, M = atom_nbrs_idx_.shape
        # [N, M, atom_fea_len]
        atom_nbrs_fea = atoms_in_fea_[atom_nbrs_idx_, :]
        # v_i, v_j, u_ij concat
        total_fea = paddle.concat([atoms_in_fea_.unsqueeze(1).expand([N, M, self.atom_fea_len]),
                                   atom_nbrs_fea, atom_bonds_in_fea_], axis=2)
        total_gated_fea = self.fc(total_fea)
        total_gated_fea = self.bn1(total_gated_fea.reshape([-1, self.atom_fea_len * 2])) \
            .reshape([N, M, self.atom_fea_len * 2])
        conv_weight, conv_self = total_gated_fea.chunk(2, axis=2)
        conv_weight = self.sigmoid(conv_weight)
        conv_self = self.softplus1(conv_self)
        # [N, atom_fea_len]
        conv_sum = self.bn2(paddle.sum(conv_weight * conv_self, axis=1))
        conv_out = self.softplus2(conv_sum + atoms_in_fea_)
        return conv_out


class CrystalGraphConvNet(nn.Layer):
    def __init__(self, input_atom_fea_len_, bond_fea_len_, atom_fea_len_=64,
                 n_conv_=3, pooling_fea_len_=128, pooling_n_hidden_=1,
                 classification_=False):
        super().__init__()
        self.classification = classification_
        self.embedding = nn.Linear(input_atom_fea_len_, atom_fea_len_)
        self.convLayers = nn.LayerList()
        for _ in range(n_conv_):
            self.convLayers.append(ConvLayer(atom_fea_len_, bond_fea_len_))
        self.conv_to_fc = nn.Linear(atom_fea_len_, pooling_fea_len_)
        self.conv_to_fc_softplus = nn.Softplus()
        if pooling_n_hidden_ > 1:
            self.fcs = nn.LayerList()
            for _ in range(pooling_fea_len_-1):
                self.fcs.append(nn.Linear(pooling_fea_len_, pooling_n_hidden_))
            self.softpluses = nn.LayerList()
            for _ in range(pooling_fea_len_ - 1):
                self.softpluses.append(nn.Softplus())
        if self.classification:
            self.fc_out = nn.Linear(pooling_fea_len_, 2)
            self.logSoftmax = nn.LogSoftmax(axis=1)
            self.dropout = nn.Dropout
        else:
            self.fc_out = nn.Linear(pooling_fea_len_, 1)

    def forward(self, input_atom_in_fea, bond_in_fea, atom_nbrs_idx, crystal_atom_idx):
        """
        N: total num of atoms in the batch
        M: Max num of neighbors
        N0: total num of crystals in the batch
        input_atom_in_fea:
        bond_in_fea:
        crystal_atom_idx:
        """
        atom_in_fea = self.embedding(input_atom_in_fea)

        # for convLayer in self.convLayers:
            # atom_in_fea = convLayer(atom_in_fea, bond_in_fea, atom_nbrs_idx)

        paddle.device.cuda.synchronize(0)
        conv_nvtx = nvtx.start_range(message="convLayer", color="blue")

        atom_in_fea = self.convLayers[0](atom_in_fea, bond_in_fea, atom_nbrs_idx)

        paddle.device.cuda.synchronize(0)
        nvtx.end_range(conv_nvtx)

        crys_fea = self.pooling(atom_in_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_in_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
               atom_in_fea.data.shape[0]
        pooling_fea = [paddle.mean(atom_in_fea[idx_map], axis=0, keepdim=True)
                       for idx_map in crystal_atom_idx]
        return paddle.concat(pooling_fea, axis=0)
