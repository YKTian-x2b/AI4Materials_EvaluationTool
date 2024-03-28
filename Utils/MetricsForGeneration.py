import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.spatial.distance import pdist, cdist
from scipy.stats import wasserstein_distance
import itertools
import torch
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from pymatgen.core.composition import Composition
import smact
from smact.screening import pauling_test

from Utils.utils import StandardScaler, frac_to_cart_coords, distance_matrix_pbc
from Utils.GenerationModelConstants import CompScalerMeans, CompScalerStds, chemical_symbols


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

COV_Cutoffs = {
    'mp_20': {'struc': 0.6, 'comp': 12.},
    'carbon_24': {'struc': 1.0, 'comp': 4.},
    'perov_5': {'struc': 0.8, 'comp': 6},
}
Percentiles = {
    'mp_20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon_24': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perov_5': np.array([0.43924842, 0.61202443, 0.7364607]),
}

def get_comp_fp(atom_types_list):
    """
    提取化学成分的特征信息
    """
    comp_fps = []
    for atom_types in atom_types_list:
        elem_counter = Counter(atom_types)
        comp = Composition(elem_counter)
        try:
            comp_fp = CompFP.featurize(comp)
        except:
            comp_fp = None
        comp_fps.append(comp_fp)
    return comp_fps


def get_structure_fp(structure_list):
    """
    提取化学结构的特征信息
    """
    structure_fps = []
    for structure in structure_list:
        if structure is None:
            struct_fp = None
            structure_fps.append(struct_fp)
            continue

        try:
            site_fps = [CrystalNNFP.featurize(structure, i) for i in range(len(structure))]
            struct_fp = np.array(site_fps).mean(axis=0)
        except:
            struct_fp = None
        structure_fps.append(struct_fp)
    return structure_fps


def getCOV(gen_atom_types_list, gt_atom_types_list, gen_structure_list, gt_structure_list, dataset_name):
    """
    有多少生成的材料是高质量的；有多少真实材料被正确预测；
    :param gen_atom_types_list: [numGen, dim_atomType] example:[1000, 5] 生成的原子类型列表
    :param gt_atom_types_list: [datasetSize, dim_atomType]  example:[3785, 5] 数据集原子类型列表
    :param gen_structure_list: [datasetSize] 元素是Structure
    :param gt_structure_list:  [numGen] 元素是Structure
    :param dataset_name: ['mp_20', 'carbon_24', 'perov_5']
    :return 覆盖率-召回率 覆盖率-精度
    """
    assert len(gen_atom_types_list) == len(gen_structure_list)
    assert len(gt_atom_types_list) == len(gt_structure_list)
    gen_comp_fps = get_comp_fp(gen_atom_types_list)
    gt_comp_fps = get_comp_fp(gt_atom_types_list)
    gen_structure_fps = get_structure_fp(gen_structure_list)
    gt_structure_fps = get_structure_fp(gt_structure_list)
    num_gen_crystals = len(gen_comp_fps)

    filtered_gen_comp_fps, filtered_gen_structure_fps = [], []
    for comp_fp, structure_fp in zip(gen_comp_fps, gen_structure_fps):
        if comp_fp is not None and structure_fp is not None:
            filtered_gen_comp_fps.append(comp_fp)
            filtered_gen_structure_fps.append(structure_fp)
    print(len(filtered_gen_comp_fps))
    CompScaler = StandardScaler(means=np.array(CompScalerMeans), stds=np.array(CompScalerStds), replace_nan_token=0.)
    gen_comp_fps = CompScaler.transform(filtered_gen_comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)
    gen_structure_fps = np.array(filtered_gen_structure_fps)
    gt_structure_fps = np.array(gt_structure_fps)

    comp_dist = cdist(gen_comp_fps, gt_comp_fps)
    structure_dist = cdist(gen_structure_fps, gt_structure_fps)

    structure_recall_dist = structure_dist.min(axis=0)
    structure_precision_dist = structure_dist.min(axis=1)
    comp_recall_dist = comp_dist.min(axis=0)
    comp_precision_dist = comp_dist.min(axis=1)

    comp_cutoff = COV_Cutoffs[dataset_name]['comp']
    structure_cutoff = COV_Cutoffs[dataset_name]['struc']
    # P表示有多少生成的材料是高质量的
    cov_precision = np.sum(np.logical_and(
        structure_precision_dist <= structure_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals
    # R表示有多少真实材料被正确预测
    cov_recall = np.mean(np.logical_and(
        structure_recall_dist <= structure_cutoff,
        comp_recall_dist <= comp_cutoff))
    return cov_precision, cov_recall


##
def getSMACTValidity(atom_types_list, use_pauling_test=True, include_alloys=True):
    """
    根据SMACT计算，总电荷呈电中性则有效，返回有效率
    :param atom_types_list: [numGen, dim_atomType] example:[1000, 5] 生成的原子类型列表
    :return [numGen, bool], 有效率
    """
    is_valid, num_valid = [], 0
    for atom_types in tqdm(atom_types_list):
        elem_counter = Counter(atom_types)
        composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        comp, count = list(zip(*composition))
        count = np.array(count)
        count = count / np.gcd.reduce(count)
        count = tuple(count.astype('int').tolist())

        elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
        space = smact.element_dictionary(elem_symbols)
        smact_elems = [e[1] for e in space.items()]
        electronegs = [e.pauling_eneg for e in smact_elems]
        ox_combos = [e.oxidation_states for e in smact_elems]
        if len(set(elem_symbols)) == 1:
            is_valid.append(True)
            num_valid += 1
            continue
        if include_alloys:
            is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
            if all(is_metal_list):
                is_valid.append(True)
                num_valid += 1
                continue

        threshold = np.max(count)
        compositions = []
        for ox_states in itertools.product(*ox_combos):
            stoichs = [(c,) for c in count]
            # Test for charge balance
            cn_e, cn_r = smact.neutral_ratios(
                ox_states, stoichs=stoichs, threshold=threshold)
            # Electronegativity test
            if cn_e:
                if use_pauling_test:
                    try:
                        electroneg_OK = pauling_test(ox_states, electronegs)
                    except TypeError:
                        # if no electronegativity dataConfig, assume it is okay
                        electroneg_OK = True
                else:
                    electroneg_OK = True
                if electroneg_OK:
                    for ratio in cn_r:
                        compositions.append(
                            tuple([elem_symbols, ox_states, ratio]))
        compositions = [(i[0], i[2]) for i in compositions]
        compositions = list(set(compositions))
        if len(compositions) > 0:
            is_valid.append(True)
            num_valid += 1
        else:
            is_valid.append(False)

    return is_valid, num_valid / len(atom_types_list)


def getStructureValidity(atom_types_list, lengths_list, angles_list, frac_coords_list, structure_list, cutoff=0.5):
    is_valid, num_valid = [], 0

    for i in tqdm(range(len(atom_types_list))):
        if structure_list[i] is None:
            is_valid.append(False)
            continue

        length = torch.from_numpy(lengths_list[i]).view(1, -1)
        angle = torch.from_numpy(angles_list[i]).view(1, -1)
        frac_coords = torch.from_numpy(frac_coords_list[i])
        num_atom = len(atom_types_list[i])
        cart_coord = frac_to_cart_coords(frac_coords, length, angle, torch.tensor([num_atom]))
        dist_mat = distance_matrix_pbc(cart_coord, length, angle)

        dist_mat += torch.diag(torch.ones([num_atom]) * cutoff)
        min_dist = dist_mat.min()
        if min_dist >= cutoff:
            is_valid.append(True)
            num_valid += 1
        else:
            is_valid.append(False)

    return is_valid, num_valid / len(is_valid)


def getElemTypeNumEMD(gen_atom_types_list, gt_atom_types_list):
    """
    生成材料元素类型数量和真实材料元素类型数量间的推土距离(EMD)
    """
    gt_elem_type_nums = []
    gen_elem_type_nums = []
    for gt_atom_types in gt_atom_types_list:
        gt_elem_type_nums.append(len(set(gt_atom_types)))
    for gen_atom_types in gen_atom_types_list:
        gen_elem_type_nums.append(len(set(gen_atom_types)))
    return wasserstein_distance(gen_elem_type_nums, gt_elem_type_nums)


def getDensityEMD(gen_structure_list, gt_structure_list):
    """
    density EMD
    """
    gen_densities = [gen_structure.density for gen_structure in gen_structure_list if gen_structure is not None]
    gt_densities = [gt_structure.density for gt_structure in gt_structure_list if gt_structure is not None]
    return wasserstein_distance(gen_densities, gt_densities)


def getSR(best_props, dataset_name):
    """
    成功率
    """
    # valid_indices = np.logical_and(getSMACTValidity(), getStructureValidity())
    # valid_indices = valid_indices.reshape(step_opt, num_opt)
    # valid_x, valid_y = valid_indices.nonzero()
    # gen_props = prop_model_eval(dataset_name, valid_idx)
    # props[valid_x, valid_y] = gen_props
    # best_props = props.min(axis=0)
    percentiles = Percentiles[dataset_name]
    sr_5 = (best_props <= percentiles[0]).mean()
    sr_10 = (best_props <= percentiles[1]).mean()
    sr_15 = (best_props <= percentiles[2]).mean()
    return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}