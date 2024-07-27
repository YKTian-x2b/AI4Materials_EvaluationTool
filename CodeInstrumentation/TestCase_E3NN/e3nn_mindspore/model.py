from mindspore import nn
import mindspore as ms
import mindspore.dataset as ds
from mindchemistry.e3.nn import FullyConnectedNet, Scatter, Gate, soft_one_hot_linspace
from mindchemistry.e3.o3 import Irrep, Irreps, TensorProduct, spherical_harmonics, FullyConnectedTensorProduct
from mindchemistry.cell.message_passing import Compose
from utils import msDataset
import time
import math
import nvtx

dtype = ms.float32


class BandLoss(nn.LossBase):
    def __init__(self, reduction="mean"):
        super().__init__(reduction)

    def construct(self, input, target):
        abs_res = ms.ops.abs(input - target)
        max_res, _ = ms.ops.max(ms.ops.abs(target))
        midres = ms.ops.pow(abs_res/max_res, 2.0)
        return ms.ops.sum(midres) / ms.ops.numel(target)


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


# 其实并不理解自定义它的用处
class CustomCompose(nn.Cell):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def construct(self, *input):
        x = self.first(*input)

        # input[0].asnumpy()
        # Gate_nvtx = nvtx.start_range(message="Gate_nvtx", color="blue")
        x = self.second(x)
        # x.asnumpy()
        # nvtx.end_range(Gate_nvtx)

        # self.second_out = x.clone()
        return x


class GraphConvolution(nn.Cell):
    def __init__(self,
                 irreps_in,
                 irreps_node_attr,
                 irreps_edge_attr,
                 irreps_out,
                 number_of_basis,
                 radial_layers,
                 radial_neurons):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_out = Irreps(irreps_out)

        self.linear_input = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in, dtype)
        self.linear_mask = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out, dtype)

        irreps_mid = []
        instructions = []
        for i, (mul, irrep_in) in enumerate(self.irreps_in):
            for j, (_, irrep_edge_attr) in enumerate(self.irreps_edge_attr):
                for irrep_mid in irrep_in * irrep_edge_attr:
                    if irrep_mid in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, irrep_mid))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [(i_1, i_2, p[i_out], mode, train) for (i_1, i_2, i_out, mode, train) in instructions]

        self.tensor_edge = TensorProduct(self.irreps_in,
                                         self.irreps_edge_attr,
                                         irreps_mid,
                                         instructions,
                                         dtype=dtype,
                                         weight_mode="custom",
                                         ncon_dtype=dtype)

        self.edge2weight = FullyConnectedNet(
            [number_of_basis] + radial_layers * [radial_neurons] + [self.tensor_edge.weight_numel], ms.ops.silu,
            dtype=dtype)

        self.linear_output = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out, dtype)
        self.scatter = Scatter()

    def construct(self,
                  node_input,
                  node_attr,
                  node_deg,
                  edge_src,
                  edge_dst,
                  edge_attr,
                  edge_length_embedded,
                  numb, n):
        node_input_features = self.linear_input(node_input, node_attr)
        node_features = ms.ops.div(node_input_features, ms.ops.pow(node_deg, 0.5))
        node_mask = self.linear_mask(node_input, node_attr)

        # node_mask.asnumpy()
        # FullyConnectedNet_nvtx = nvtx.start_range(message="FullyConnectedNet", color="blue")

        edge_weight = self.edge2weight(edge_length_embedded)

        # edge_weight.asnumpy()
        # nvtx.end_range(FullyConnectedNet_nvtx)

        # ms.hal.synchronize() 2.3版本特性 目前不支持GPU
        # edge_weight.asnumpy()
        # TensorProduct_nvtx = nvtx.start_range(message="TensorProduct", color="blue")
        edge_features = self.tensor_edge(node_features[edge_src], edge_attr, edge_weight)
        # ms.hal.synchronize()
        # edge_features.asnumpy()
        # nvtx.end_range(TensorProduct_nvtx)

        edge_features.asnumpy()
        Scatter_nvtx = nvtx.start_range(message="Scatter", color="blue")

        node_features = self.scatter(edge_features, edge_dst, dim_size=node_features.shape[0])

        node_features.asnumpy()
        nvtx.end_range(Scatter_nvtx)


        node_features = ms.ops.div(node_features, ms.ops.pow(node_deg, 0.5))
        node_output_features = self.linear_output(node_features, node_attr)

        node_output = node_output_features

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        mask = self.linear_mask.output_mask
        c_x = (1 - mask) + c_x * mask
        return c_s * node_mask + c_x * node_output


class GraphNetworkVVN(nn.Cell):
    def __init__(self,
                 mul,
                 irreps_out,
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,
                 node_embed_dim,
                 input_dim,
                 input_embed_dim):
        super().__init__()
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim) + 'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim) + 'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: ms.ops.silu, -1: ms.ops.tanh}
        act_gates = {1: ms.ops.sigmoid, -1: ms.ops.tanh}

        self.layers = nn.CellList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if
                                     ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if
                                   ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps_in, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated, dtype, dtype)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)
            irreps_in = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))
        self.layers.append(GraphConvolution(irreps_in,
                                            self.irreps_node_attr,
                                            self.irreps_edge_attr,
                                            self.irreps_out,
                                            number_of_basis,
                                            radial_layers,
                                            radial_neurons))
        self.emx = nn.Dense(input_dim, input_embed_dim, dtype=dtype)
        self.emz = nn.Dense(node_dim, node_embed_dim, dtype=dtype)

    def construct(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis,
                                                     basis='gaussian', cutoff=False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_attr = edge_sh
        numb = data['numb']
        x = ms.ops.relu(self.emx(ms.ops.relu(data['x'])))
        z = ms.ops.relu(self.emz(ms.ops.relu(data['z'])))
        node_deg = data['node_deg']
        n = None
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n)
        x = x.reshape((1, -1))[:, numb:]
        return x


