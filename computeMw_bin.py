"""
Compute Mw, M0, total fault area, slipped area (slip > threshold) and
average slip from a SeisSol fault XDMF output (works for both .h5 and .bin
backends via seissolxdmf).

Usage:
    python computeMw.py <fault.xdmf> [--slip-threshold 0.03]
"""

import argparse
import numpy as np
import seissolxdmf
from math import log10

# --- 1-D rigidity profile: linear interpolation between nodes ---
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
    """Piecewise-linear mu(z), vectorized. z in meters (negative = depth)."""
    return np.interp(z, MU_NODES_Z, MU_NODES_MU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xdmf", help="SeisSol fault XDMF file")
    ap.add_argument("--slip-threshold", type=float, default=0.03,
                    help="Slip threshold for slipped-area stats (m, default 0.03)")
    args = ap.parse_args()

    sx = seissolxdmf.seissolxdmf(args.xdmf)
    geom = sx.ReadGeometry()              # (nNodes, 3)
    conn = sx.ReadConnect()               # (nElements, 3)
    ndt = sx.ReadNdt()
    available = set(sx.ReadAvailableDataFields())
    print(f"using time step {ndt - 1} (0-indexed last of {ndt})")
    print(f"available fields: {sorted(available)}")

    if "ASl" in available:
        slip = sx.ReadData("ASl", ndt - 1)
    elif {"Sls", "Sld"}.issubset(available):
        Sls = sx.ReadData("Sls", ndt - 1)
        Sld = sx.ReadData("Sld", ndt - 1)
        slip = np.sqrt(Sls ** 2 + Sld ** 2)
    else:
        raise SystemExit("ASl or (Sls, Sld) not available in XDMF")

    # Triangle areas (vectorized)
    v0 = geom[conn[:, 0]]
    v1 = geom[conn[:, 1]]
    v2 = geom[conn[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Centroid z for rigidity lookup
    z_centroid = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    mu = mu_at(z_centroid)

    # Moments and area stats
    subfault_moment = mu * slip * area
    M0 = float(subfault_moment.sum())
    total_area = float(area.sum())

    mask = slip > args.slip_threshold
    slipped_area = float(area[mask].sum())
    avg_slip = float((slip[mask] * area[mask]).sum() / slipped_area) if slipped_area > 0 else 0.0

    Mw = 2.0 * log10(M0) / 3.0 - 6.07

    print("M0              = %.6e N.m" % M0)
    print("Mw              = %.4f" % Mw)
    print("Total fault area= %.3f km^2" % (total_area / 1e6))
    print("Slipped area    = %.3f km^2 (slip > %.3f m)" % (slipped_area / 1e6, args.slip_threshold))
    print("Average slip    = %.3f m (area-weighted, over slipped region)" % avg_slip)


if __name__ == "__main__":
    main()
