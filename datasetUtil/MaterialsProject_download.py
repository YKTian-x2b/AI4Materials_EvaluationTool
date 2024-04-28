import sys
import os
import numpy as nps
import csv
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester
from mp_api.client.core.utils import validate_ids

from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


def get_cif(src_path, dst_path, id_prop_path, mpr):
    validate_material_ids = []
    with open(src_path) as src_file:
        src_reader = csv.reader(src_file)
        for row in src_reader:
            validate_material_ids.append(row[0])
    validate_material_ids = validate_ids(validate_material_ids)

    id_prop_file = open(id_prop_path, 'w')
    id_prop_writer = csv.writer(id_prop_file)
    docs = mpr.summary.search(
        material_ids=validate_material_ids, fields=["material_id", "formation_energy_per_atom", "structure"]
    )
    for doc in docs:
        try:
            mat_id = doc.material_id
            cif_writer = CifWriter(doc.structure)
            cif_writer.write_file(dst_path + mat_id[3:] + ".cif")
            id_prop_writer.writerow((mat_id[3:], doc.formation_energy_per_atom))
        except AttributeError as ae:
            print(ae)
            continue

    id_prop_file.close()


def MP_download():
    API_KEY = "ZgkExtDvolhh3FhHT2FAibb2Jo0JhmFL"

    src_path_base = current_dir + 'material-data/'
    dst_path = current_dir + 'root_dir/'
    id_prop_path = current_dir + 'root_dir/id_prop.csv'
    with MPRester(API_KEY) as mpr:
        # get_cif(src_path_base + "mp-ids-demo.csv", dst_path, id_prop_path, mpr)
        get_cif(src_path_base + "mp-ids-3402.csv", dst_path, id_prop_path, mpr)
        # get_cif(src_path_base + 'mp-ids-27430.csv', dst_path, id_prop_path, mpr)
        # get_cif(rc_path_base + 'mp-ids-46744.csv', dst_path, id_prop_path, mpr)


if __name__ == '__main__':
    # MP_download()
    directory = current_dir + 'root_dir/'
    files_and_dirs = os.listdir(directory)
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
    print(len(files))


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
