import sys
projectPath = "/opt/AI4Sci/AI4S_EvaluationToolset/AI4Material_EvaluationTool/CGCNN/cgcnn-master/"
sys.path.append(projectPath)
import numpy as np
import csv
import torch
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from ase.visualize import view

from cgcnn import data


def get_cif(src_path, materials_id, mpr):
    with open(src_path) as src_file:
        src_reader = csv.reader(src_file)
        for row in src_reader:
            materials_id.append(row[0])
            structure = mpr.get_structure_by_material_id(row[0])
            mp_id_str = row[0][3:]
            cif_writer = CifWriter(structure)
            cif_writer.write_file(datasetPath + mp_id_str + ".cif")


def MP_download():
    API_KEY = "ZgkExtDvolhh3FhHT2FAibb2Jo0JhmFL"
    mpr = MPRester(API_KEY)

    materials_id = []
    # get cif
    get_cif(projectPath + '/data/material-data/mp-ids-3402.csv', materials_id, mpr)
    get_cif(projectPath + '/data/material-data/mp-ids-27430.csv', materials_id, mpr)
    get_cif(projectPath + '/data/material-data/mp-ids-46744.csv', materials_id, mpr)

    # get id_prop.csv
    with open(projectPath + '/root_dir/id_prop.csv', 'w') as id_prop_file:
        id_prop_writer = csv.writer(id_prop_file)
        docs = mpr.materials.summary.search(
            material_ids=materials_id, fields=["material_id", "band_gap"]
        )
        for doc in docs:
            id_prop_writer.writerow((doc.material_id, doc.band_gap))


def visualize(datasetPath):
    crystal = Structure.from_file(datasetPath + "7090.cif")
    Converter = AseAtomsAdaptor()
    ase_crystal = Converter.get_atoms(crystal)
    # view(ase_crystal)
    # print(crystal.sites)
    r = 3
    all_nbrs = crystal.get_all_neighbors(3)
    neighbors_from_atom_0 = all_nbrs[0]
    print(all_nbrs)


def tmp():
    print('tmp')


if __name__ == '__main__':
    datasetPath = projectPath + 'root_dir/'
    MP_download()

    # visualize(datasetPath)

    # dataset = data.CIFData(projectPath + "root_dir")
    # structures, _, _ = dataset[0]
    # orig_atom_fea_len = structures[0].shape[-1]
    # nbr_fea_len = structures[1].shape[-1]
    # print(structures)
