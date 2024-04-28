# import paddle
# from paddle import nn
import numpy as np
import os
from pymatgen.core.structure import Structure

if __name__ == "__main__":
    root_dir = '../CGCNN/CGCNN_paddle/root_dir'
    cif_id = 'mp-7090'
    crystal = Structure.from_file(os.path.join(root_dir, cif_id + '.cif'))
    # print(crystal)
    # print(len(crystal))
    # for i in range(len(crystal)):
    #     print(crystal[i])

    all_nbrs = crystal.get_all_neighbors(8, include_index=True)
    # all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    # nbr_fea_idx, nbr_fea = [], []
    # for nbr in all_nbrs:
    #     if len(nbr) < 12:
    #         nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
    #                            [0] * (12 - len(nbr)))
    #         nbr_fea.append(list(map(lambda x: x[1], nbr)) +
    #                        [8 + 1.] * (12 - len(nbr)))
    #     else:
    #         nbr_fea_idx.append(list(map(lambda x: x[2],
    #                                     nbr[:12])))
    #         nbr_fea.append(list(map(lambda x: x[1],
    #                                 nbr[:12])))
    #
    # print(nbr_fea_idx)
    # print(nbr_fea)
    print(all_nbrs[0][0])

    for jjj in range(len(all_nbrs[0][0])):
        print(all_nbrs[0][0][jjj])
    # dmin = 0
    # dmax = 8
    # step = 0.2
    # filter = np.arange(dmin, dmax + step, step)
    # var = step
    # distances = []
    #
    # np.exp(-(distances[..., np.newaxis] - filter) ** 2 / var ** 2)