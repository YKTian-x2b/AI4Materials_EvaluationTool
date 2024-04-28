# Full Formula (Na2 Mg2 Sb2)
# Reduced Formula: NaMgSb
# abc   :   4.649156   4.649156   7.615181
# angles:  90.000000  90.000000  90.000000
# Sites (6)
#   #  SP       a     b         c
# ---  ----  ----  ----  --------
#   0  Na    0.25  0.25  0.357428
#   1  Na    0.75  0.75  0.642572
#   2  Mg    0.25  0.75  0
#   3  Mg    0.75  0.25  0
#   4  Sb    0.25  0.25  0.774088
#   5  Sb    0.75  0.75  0.225912

# 6

# [1.16228894 1.16228894 2.72187576] Na
# [3.48686682 3.48686682 4.89330557] Na
# [1.16228894e+00 3.48686682e+00 2.84678686e-16] Mg
# [3.48686682e+00 1.16228894e+00 2.84678686e-16] Mg
# [1.16228894 1.16228894 5.89482384] Sb
# [3.48686682 3.48686682 1.72035749] Sb


import os
from pymatgen.core.structure import Structure

if __name__ == "__main__":
    root_dir = './root_dir'
    cif_id = 'mp-7090'
    crystal = Structure.from_file(os.path.join(root_dir, cif_id + '.cif'))
    print(crystal)
    print(len(crystal))
    for i in range(len(crystal)):
        print(crystal[i])
