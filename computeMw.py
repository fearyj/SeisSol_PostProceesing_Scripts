import h5py
import numpy as np
from math import sqrt, log10, pow
import sys

if len(sys.argv) != 2:
    print("usage: computeMw.py file.h5")
    exit()

# 1-D rigidity profile: linear interpolation between nodes.
# z in meters (negative = depth), mu in Pa. Sorted ascending in z.
MU_NODES_Z = np.array([
    -400000.0, -42000.0, -38000.0, -34000.0, -32000.0, -28000.0, -24000.0,
    -20000.0, -16000.0, -12000.0, -8000.0, -2000.0, 0.0, 5000.0,
])
MU_NODES_MU = np.array([
    7.116606e10, 5.454917e10, 5.122235e10, 5.029105e10, 4.665131e10,
    3.615821e10, 3.479571e10, 3.402888e10, 3.308237e10, 3.014323e10,
    2.925279e10, 2.872493e10, 2.063483e10, 2.063483e10,
])


def mu_at(z):
    return float(np.interp(z, MU_NODES_Z, MU_NODES_MU))


h5f = h5py.File(sys.argv[1], "r")

xyz = h5f["geometry"][:, :]
tetra = h5f["connect"][:, :]

ndt = h5f["SRs"].shape[0]
print("using %d th time step " % ndt)

if "ASl" in h5f:
    use_ASl = True
    ASl = h5f["ASl"][ndt - 1, :]
elif ("Sls" in h5f) & ("Sld" in h5f):
    use_ASl = False
    Sls = h5f["Sls"][ndt - 1, :]
    Sld = h5f["Sld"][ndt - 1, :]
else:
    print("ASl or Sls,Sld could not be found in file")
    exit()

nElements = tetra.shape[0]

slip_threshold = 0.03

M0 = 0.0
total_area = 0.0
slipped_area = 0.0
slip_area_sum = 0.0
for i in range(nElements):
    if (i != 0) and (i % max(nElements // 100, 1) == 0):
        print("done %d" % (i * 100 // nElements))
    area = 0.5 * np.linalg.norm(
        np.cross(
            xyz[tetra[i, 1]] - xyz[tetra[i, 0]], xyz[tetra[i, 2]] - xyz[tetra[i, 0]]
        )
    )
    z = (xyz[tetra[i, 0], 2] + xyz[tetra[i, 1], 2] + xyz[tetra[i, 2], 2]) / 3.0
    if use_ASl:
        slip = ASl[i]
    else:
        slip = sqrt(pow(Sls[i], 2) + pow(Sld[i], 2))
    mu = mu_at(z)
    M0 = M0 + mu * slip * area
    total_area += area
    if slip > slip_threshold:
        slipped_area += area
        slip_area_sum += slip * area

avg_slip = slip_area_sum / slipped_area if slipped_area > 0 else 0.0
Mw = 2.0 * log10(M0) / 3.0 - 6.07

print("M0              = %.6e N.m" % M0)
print("Mw              = %.4f" % Mw)
print("Total fault area= %.3f km^2" % (total_area / 1e6))
print("Slipped area    = %.3f km^2 (slip > %.3f m)" % (slipped_area / 1e6, slip_threshold))
print("Average slip    = %.3f m (area-weighted, over slipped region)" % avg_slip)
