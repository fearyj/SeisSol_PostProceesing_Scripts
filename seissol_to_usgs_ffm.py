"""
Convert SeisSol fault XDMF output to a USGS finite-fault-model (FFM) CSV.

Usage:
    python seissol_to_usgs_ffm.py <fault.xdmf> <scenario_fault.yaml> <out.csv>
                                  [--rise ide|threshold|p5_p95]

Output columns (one row per fault triangle):
    lat, lon        centroid coordinates (WGS84, deg)
    x_ew, y_ns      centroid position relative to hypocenter (km, E+ / N+)
    z_depth         centroid depth (km, positive down)
    slip            net slip magnitude sqrt(Sls^2 + Sld^2) at final step (m)
    rake            atan2(-Sld, Sls) in degrees (Aki-Richards convention)
    trup            rupture time from RT variable (s)
    trise           rise time (s); Ide (2002) effective rise time by default
    sf_moment       subfault seismic moment = mu * slip * area (N.m)

Hypocenter is parsed from the scenario YAML's `forced_rupture_time` block,
specifically from the line  r = sqrt(pow(x-A, 2.0) + pow(y+B, 2.0) + pow(z+C, 2.0));
with the sign flipped:   x_hypo = +A,   y_hypo = -B,   z_hypo = -C  (meters, TM).
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
import seissolxdmf
from pyproj import Transformer

# --- SeisSol projection (constant for the Mentawai setup) ---
SEISSOL_PROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lat_0=0 +lon_0=100"

# --- 1-D rigidity profile: linear interpolation between nodes ---
# Nodes: (z [m, negative = depth], mu [Pa]). Sorted ascending in z below.
MU_NODES_Z = np.array([
    -400000.0, -42000.0, -38000.0, -34000.0, -32000.0, -28000.0, -24000.0,
    -20000.0, -16000.0, -12000.0, -8000.0, -2000.0, 0.0, 5000.0,
])
MU_NODES_MU = np.array([
    7.116606e10, 5.454917e10, 5.122235e10, 5.029105e10, 4.665131e10,
    3.615821e10, 3.479571e10, 3.402888e10, 3.308237e10, 3.014323e10,
    2.925279e10, 2.872493e10, 2.063483e10, 2.063483e10,
])

# Rise-time computation parameters
RT_THRESHOLD_FRAC = 0.05
RT_SR_FLOOR = 1e-4  # m/s


def mu_avg(z):
    """Piecewise-linear mu(z) from the node table. z in meters (negative = depth)."""
    return np.interp(z, MU_NODES_Z, MU_NODES_MU)


def parse_hypocenter(yaml_path):
    """Extract hypocenter (x, y, z) in meters from the forced_rupture_time block."""
    with open(yaml_path) as f:
        text = f.read()

    coords = {}
    for var in ("x", "y", "z"):
        m = re.search(rf"pow\(\s*{var}\s*([+\-])\s*([0-9.eE+\-]+)\s*,", text)
        if m is None:
            raise ValueError(f"Could not find pow({var}...) in {yaml_path}")
        sign, num = m.group(1), float(m.group(2))
        coords[var] = num if sign == "-" else -num
    return coords["x"], coords["y"], coords["z"]


def compute_rise_time(SRs, times, slip, method):
    ndt, nE = SRs.shape
    peak_sr = SRs.max(axis=0)
    slipped = peak_sr > RT_SR_FLOOR

    if method == "ide":
        int_sr2 = np.trapezoid(SRs ** 2, times, axis=0)
        trise = np.zeros(nE)
        ok = slipped & (int_sr2 > 0) & (slip > 0)
        trise[ok] = slip[ok] ** 2 / int_sr2[ok]
    elif method == "threshold":
        thr = RT_THRESHOLD_FRAC * np.maximum(peak_sr, RT_SR_FLOOR)
        active = SRs > thr[None, :]
        idx_first = np.argmax(active, axis=0)
        idx_last = (ndt - 1) - np.argmax(active[::-1], axis=0)
        trise = np.where(slipped, times[idx_last] - times[idx_first], 0.0)
    elif method == "p5_p95":
        dt = np.diff(times, prepend=times[0])
        cum = np.cumsum(SRs * dt[:, None], axis=0)
        final = cum[-1]
        trise = np.zeros(nE)
        for j in np.where(slipped & (final > 0))[0]:
            c = cum[:, j] / final[j]
            i5 = min(np.searchsorted(c, 0.05), ndt - 1)
            i95 = min(np.searchsorted(c, 0.95), ndt - 1)
            trise[j] = times[i95] - times[i5]
    else:
        raise ValueError(f"Unknown rise-time method: {method}")
    return trise, slipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xdmf", help="SeisSol fault XDMF file")
    ap.add_argument("yaml", help="Scenario fault YAML (for hypocenter)")
    ap.add_argument("out_csv", help="Output USGS FFM CSV path")
    ap.add_argument("--rise", choices=["ide", "threshold", "p5_p95"],
                    default="ide", help="Rise-time definition (default: ide)")
    args = ap.parse_args()

    # --- Hypocenter from YAML ---
    hx, hy, hz = parse_hypocenter(args.yaml)
    fwd = Transformer.from_crs(SEISSOL_PROJ, "EPSG:4326", always_xy=True)
    hlon, hlat = fwd.transform(hx, hy)
    print(f"Hypocenter (TM m) : x={hx:.1f}, y={hy:.1f}, z={hz:.1f}")
    print(f"Hypocenter (WGS84): lon={hlon:.4f}, lat={hlat:.4f}, depth={-hz/1000:.3f} km")

    # --- Read SeisSol ---
    print(f"Reading {args.xdmf} ...")
    sx = seissolxdmf.seissolxdmf(args.xdmf)
    geom = sx.ReadGeometry()
    conn = sx.ReadConnect()
    ndt = sx.ReadNdt()
    times = np.asarray(sx.ReadTimes())
    available = set(sx.ReadAvailableDataFields())
    print(f"  Nodes: {geom.shape[0]}, Elements: {conn.shape[0]}, Steps: {ndt}")
    print(f"  Fields: {sorted(available)}")

    Sls = sx.ReadData("Sls", ndt - 1)
    Sld = sx.ReadData("Sld", ndt - 1)
    slip = np.sqrt(Sls ** 2 + Sld ** 2)
    RT = sx.ReadData("RT", ndt - 1)

    # --- Triangle centroid / area / projection ---
    v0, v1, v2 = geom[conn[:, 0]], geom[conn[:, 1]], geom[conn[:, 2]]
    centroid = (v0 + v1 + v2) / 3.0
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    lon, lat = fwd.transform(centroid[:, 0], centroid[:, 1])
    x_ew = (centroid[:, 0] - hx) / 1000.0
    y_ns = (centroid[:, 1] - hy) / 1000.0
    z_depth = -centroid[:, 2] / 1000.0

    rake = np.degrees(np.arctan2(-Sld, Sls))

    # --- Rise time from slip-rate magnitude time series ---
    print(f"Reading SRs/SRd time series and computing rise time ({args.rise}) ...")
    SR_mag = np.empty((ndt, conn.shape[0]), dtype=np.float64)
    for k in range(ndt):
        srs = sx.ReadData("SRs", k)
        srd = sx.ReadData("SRd", k)
        SR_mag[k] = np.sqrt(srs ** 2 + srd ** 2)
    trise, slipped = compute_rise_time(SR_mag, times, slip, args.rise)

    # --- Subfault moment ---
    mu = mu_avg(centroid[:, 2])
    sf_moment = mu * slip * area

    # --- Write CSV ---
    df = pd.DataFrame({
        "lat": np.round(lat, 4),
        "lon": np.round(lon, 4),
        "x_ew": np.round(x_ew, 4),
        "y_ns": np.round(y_ns, 4),
        "z_depth": np.round(z_depth, 4),
        "slip": np.round(slip, 4),
        "rake": np.round(rake, 4),
        "trup": np.round(RT, 2),
        "trise": np.round(trise, 2),
        "sf_moment": [f"{v:.3g}" for v in sf_moment],
    })
    df.to_csv(args.out_csv, index=False)

    print(f"Wrote {args.out_csv}  ({len(df)} subfaults)")
    print(f"  Total M0 : {float(np.dot(mu, slip * area)):.3e} N.m")
    if slipped.any():
        print(f"  Rise time: min/mean/max = "
              f"{trise[slipped].min():.2f} / {trise[slipped].mean():.2f} / {trise[slipped].max():.2f} s")


if __name__ == "__main__":
    main()
