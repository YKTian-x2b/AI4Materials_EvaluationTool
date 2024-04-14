import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the dataConfig :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the dataConfig by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed dataConfig with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed dataConfig with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)
    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def frac_to_cart_coords(frac_coords, lengths, angles, num_atoms,):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    cart_coords = torch.einsum('bi,bij->bj', frac_coords.float(), lattice_nodes.float())  # cart coords
    return cart_coords


OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]


def distance_matrix_pbc(cart_coords, lengths, angles):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each dataConfig point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    num_atoms = cart_coords.shape[0]

    unit_cell = torch.tensor(OFFSET_LIST, device=cart_coords.device).float()
    num_cells = len(unit_cell)
    unit_cell = torch.transpose(unit_cell, 0, 1).view(1, 3, num_cells)

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = cart_coords.view(-1, 1, 3, 1).expand(-1, -1, -1, num_cells)
    pos2 = cart_coords.view(1, -1, 3, 1).expand(-1, -1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distances_pbc = (pos1 - pos2).norm(dim=-2)
    atom_distances, _ = atom_distances_pbc.min(dim=-1)

    return atom_distances


def readCSV(filePath):
    with open(filePath) as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row for row in reader]
    print({'duration (s)' : column[0]['resources/duration (s)']})
    print({'gpu:0/gpu_utilization (%)/mean': column[0]['resources/gpu:0/gpu_utilization (%)/mean']})
    print({'gpu:0/memory_utilization (%)/mean': column[0]['resources/gpu:0/memory_utilization (%)/mean']})
    print({'gpu:0/power_usage (W)/mean': column[0]['resources/gpu:0/power_usage (W)/mean']})
    print({'host/cpu_percent (%)/mean': column[0]['resources/host/cpu_percent (%)/mean']})
    print({'host/memory_percent (%)/mean': column[0]['resources/host/memory_percent (%)/mean']})


"""
metrics-daemon/host/cpu_percent (%)/mean,
metrics-daemon/host/cpu_percent (%)/min,
metrics-daemon/host/cpu_percent (%)/max,
metrics-daemon/host/cpu_percent (%)/last,
metrics-daemon/host/memory_percent (%)/mean,
metrics-daemon/host/memory_percent (%)/min,
metrics-daemon/host/memory_percent (%)/max,
metrics-daemon/host/memory_percent (%)/last,
metrics-daemon/host/swap_percent (%)/mean,
metrics-daemon/host/swap_percent (%)/min,
metrics-daemon/host/swap_percent (%)/max,
metrics-daemon/host/swap_percent (%)/last,
metrics-daemon/host/memory_used (GiB)/mean,
metrics-daemon/host/memory_used (GiB)/min,
metrics-daemon/host/memory_used (GiB)/max,
metrics-daemon/host/memory_used (GiB)/last,
metrics-daemon/host/load_average (%) (1 min)/mean,
metrics-daemon/host/load_average (%) (1 min)/min,
metrics-daemon/host/load_average (%) (1 min)/max,
metrics-daemon/host/load_average (%) (1 min)/last,
metrics-daemon/host/load_average (%) (5 min)/mean,
metrics-daemon/host/load_average (%) (5 min)/min,
metrics-daemon/host/load_average (%) (5 min)/max,
metrics-daemon/host/load_average (%) (5 min)/last,
metrics-daemon/host/load_average (%) (15 min)/mean,
metrics-daemon/host/load_average (%) (15 min)/min,
metrics-daemon/host/load_average (%) (15 min)/max,
metrics-daemon/host/load_average (%) (15 min)/last,
metrics-daemon/gpu:0/memory_used (MiB)/mean,
metrics-daemon/gpu:0/memory_used (MiB)/min,
metrics-daemon/gpu:0/memory_used (MiB)/max,
metrics-daemon/gpu:0/memory_used (MiB)/last,
metrics-daemon/gpu:0/memory_free (MiB)/mean,
metrics-daemon/gpu:0/memory_free (MiB)/min,
metrics-daemon/gpu:0/memory_free (MiB)/max,
metrics-daemon/gpu:0/memory_free (MiB)/last,
metrics-daemon/gpu:0/memory_total (MiB)/mean,
metrics-daemon/gpu:0/memory_total (MiB)/min,
metrics-daemon/gpu:0/memory_total (MiB)/max,
metrics-daemon/gpu:0/memory_total (MiB)/last,
metrics-daemon/gpu:0/memory_percent (%)/mean,
metrics-daemon/gpu:0/memory_percent (%)/min,
metrics-daemon/gpu:0/memory_percent (%)/max,
metrics-daemon/gpu:0/memory_percent (%)/last,
metrics-daemon/gpu:0/gpu_utilization (%)/mean,
metrics-daemon/gpu:0/gpu_utilization (%)/min,
metrics-daemon/gpu:0/gpu_utilization (%)/max,
metrics-daemon/gpu:0/gpu_utilization (%)/last,
metrics-daemon/gpu:0/memory_utilization (%)/mean,
metrics-daemon/gpu:0/memory_utilization (%)/min,
metrics-daemon/gpu:0/memory_utilization (%)/max,
metrics-daemon/gpu:0/memory_utilization (%)/last,
metrics-daemon/gpu:0/fan_speed (%)/mean,
metrics-daemon/gpu:0/fan_speed (%)/min,
metrics-daemon/gpu:0/fan_speed (%)/max,
metrics-daemon/gpu:0/fan_speed (%)/last,
metrics-daemon/gpu:0/temperature (C)/mean,
metrics-daemon/gpu:0/temperature (C)/min,
metrics-daemon/gpu:0/temperature (C)/max,
metrics-daemon/gpu:0/temperature (C)/last,
metrics-daemon/gpu:0/power_usage (W)/mean,
metrics-daemon/gpu:0/power_usage (W)/min,
metrics-daemon/gpu:0/power_usage (W)/max,
metrics-daemon/gpu:0/power_usage (W)/last,
metrics-daemon/duration (s),
metrics-daemon/timestamp,metrics-daemon/last_timestamp
"""
def readCSV_v2(filePath):
    with open(filePath) as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row for row in reader]
    for idx in range(len(column)):
        print('iter:', idx)
        print({'duration (s)': column[idx]['metrics-daemon/duration (s)']})
        print({'gpu_utilization (%)/mean': column[idx]['metrics-daemon/gpu:0/gpu_utilization (%)/mean']})
        print({'memory_utilization (%)/mean': column[idx]['metrics-daemon/gpu:0/memory_utilization (%)/mean']})
        print({'power_usage (W)/mean': column[idx]['metrics-daemon/gpu:0/power_usage (W)/mean']})
        print({'host/cpu_percent (%)/mean': column[idx]['metrics-daemon/host/cpu_percent (%)/mean']})
        print({'host/memory_percent (%)/mean': column[idx]['metrics-daemon/host/memory_percent (%)/mean']})


def draw(filePath):
    df = pd.read_csv(filePath)
    duration_label = 'metrics-daemon/duration (s)'
    gpu_utilization_label = 'metrics-daemon/gpu:0/gpu_utilization (%)/mean'
    gpu_memory_utilization_label = 'metrics-daemon/gpu:0/memory_utilization (%)/mean'
    host_cpu_percent_label = 'metrics-daemon/host/cpu_percent (%)/mean'
    host_memory_percent_label = 'metrics-daemon/host/memory_percent (%)/mean'
    interval = 10
    x_axis_data = df[duration_label][::interval]
    y_gpu_utilization_data = df[gpu_utilization_label][::interval]
    y_gpu_memory_utilization_data = df[gpu_memory_utilization_label][::interval]
    y_host_cpu_percent_data = df[host_cpu_percent_label][::interval]
    y_host_memory_percent_data = df[host_memory_percent_label][::interval]

    plt.ylim((0, 100))
    plt.plot(x_axis_data, y_gpu_utilization_data, marker='s', markersize=4, color='tomato',
             linestyle='-', label='gpu_utilization', alpha=0.8)
    plt.plot(x_axis_data, y_gpu_memory_utilization_data, marker='o', markersize=4, color='y',
             linestyle='-', label='gpu_memory_utilization', alpha=0.8)
    plt.plot(x_axis_data, y_host_cpu_percent_data, marker='*', markersize=4, color='m',
             linestyle='--', label='host_cpu_percent', alpha=0.8)
    plt.plot(x_axis_data, y_host_memory_percent_data, marker='x', markersize=4, color='g',
             linestyle='--', label='host_memory_percent', alpha=0.8)
    # plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.tight_layout()

    # 创建图例，并将其放置在图的右下角外部
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0.)
    plt.subplots_adjust(top=0.9)
    plt.xlabel('duration(s)')  # 数据库的单位得改一下
    plt.ylabel('utilization/precent(%)')  # y_label
    # plt.title('BlackBoxResourceUtilization')

    plt.savefig('BlackBoxResource.jpg')
    plt.show()