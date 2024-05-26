from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
import pandas as pd
import sys

# To access the Materials Project dataset via pymatgen, you will need an API key which can obtain via
# https://www.materialsproject.org/dashboard
# See https://pymatgen.org/pymatgen.ext.matproj.html for more details.

# If you have set your Materials Project API key in your ~/.pmgrc.yaml with (in the command line)

if __name__ == "__main__":
    USER_API_KEY = "ZgkExtDvolhh3FhHT2FAibb2Jo0JhmFL"

    icsd_pd = pd.read_csv('data/mp_data.csv', header=None)
    validate_material_ids = icsd_pd.iloc[:, 0].tolist()
    with MPRester(USER_API_KEY) as mpr:
        docs = mpr.summary.search(material_ids=validate_material_ids, fields=["material_id", "structure"])
        for doc in docs:
            try:
                mat_id = doc.material_id
                cif_writer = CifWriter(doc.structure)
                cif_writer.write_file('data/' + mat_id + '.cif')
            except AttributeError as ae:
                print(ae)
                continue
    print('Download complete!')
