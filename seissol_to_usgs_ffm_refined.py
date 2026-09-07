"""
seissol_to_usgs_ffm_refined.py

Two-stage conversion from SeisSol fault XDMF to a USGS FFM CSV:

  Stage 1 – Midpoint subdivision (--levels N, default 1)
    Each original fault triangle is split into 4**N equal-area sub-triangles.
    Slip, rake, trup, trise, and mu are inherited from the parent, so:
      - total M0 = sum(mu*slip*area) is preserved to machine precision
      - unweighted and area-weighted mean slip are preserved exactly

  Stage 2 – Rectangular rebinning (optional: --rect-csv PATH --rect-cell-size KM)
    Project sub-triangle centroids onto the best-fit fault plane and bin them
    onto a regular rectangular (along-strike × along-dip) grid. Properties are
    area-weighted averaged per cell; sf_moment sums the true M0 contribution
    of all triangles in the cell. Total M0 is conserved to ~machine precision.

Usage:
    python seissol_to_usgs_ffm_refined.py <fault.xdmf> <scenario.yaml> <out_tri.csv>
           [--rise ide|threshold|p5_p95]
           [--levels N]
           [--rect-csv out_rect.csv]
           [--rect-cell-size KM]          (default: 10.0 km)

Grid-size statistics (equivalent square side = sqrt(area)) are printed for
the original triangles, the refined triangles, and (if requested) the
rectangular cells.
"""

import argparse
import re

import numpy as np
import pandas as pd
import seissolxdmf
from pyproj import Transformer

# --- SeisSol projection (constant for the Mentawai setup) ---
SEISSOL_PROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lat_0=0 +lon_0=100"

# --- 1-D rigidity profile: linear interpolation between nodes ---
MU_NODES_Z = np.array([
    -400000.0, -42000.0, -38000.0, -34000.0, -32000.0, -28000.0, -24000.0,
    -20000.0, -16000.0, -12000.0, -8000.0, -2000.0, 0.0, 5000.0,
])
MU_NODES_MU = np.array([
    7.116606e10, 5.454917e10, 5.122235e10, 5.029105e10, 4.665131e10,
    3.615821e10, 3.479571e10, 3.402888e10, 3.308237e10, 3.014323e10,
    2.925279e10, 2.872493e10, 2.063483e10, 2.063483e10,
])

RT_THRESHOLD_FRAC = 0.05
RT_SR_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def mu_avg(z):
    return np.interp(z, MU_NODES_Z, MU_NODES_MU)


def parse_hypocenter(yaml_path):
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


def subdivide_triangles(v0, v1, v2, levels):
    """Midpoint-subdivide each triangle `levels` times (×4 per level).

    Returns (sv0, sv1, sv2, parent) where parent[i] is the original triangle
    index for child i.
    """
    parent = np.arange(v0.shape[0])
    for _ in range(max(levels, 0)):
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)
        v0 = np.concatenate([v0, m01, m20, m01], axis=0)
        v1 = np.concatenate([m01, v1, m12, m12], axis=0)
        v2 = np.concatenate([m20, m12, v2, m20], axis=0)
        parent = np.tile(parent, 4)
    return v0, v1, v2, parent


# ---------------------------------------------------------------------------
# Grid-size reporting
# ---------------------------------------------------------------------------

def print_grid_stats(area_m2, label):
    """Report subfault size statistics using sqrt(area) as the equivalent
    square side length — the standard measure in finite-fault seismology."""
    side_km = np.sqrt(area_m2) / 1e3
    area_km2 = area_m2 / 1e6
    print(f"  [{label}]  n={len(area_m2)}")
    print(f"    Area (km^2) : min={area_km2.min():.4f}  "
          f"mean={area_km2.mean():.4f}  max={area_km2.max():.4f}")
    print(f"    Equiv side  : min={side_km.min():.3f}  "
          f"mean={side_km.mean():.3f}  max={side_km.max():.3f} km")


# ---------------------------------------------------------------------------
# Rectangular grid
# ---------------------------------------------------------------------------

def find_fault_frame(centroids, areas):
    """Fit a plane to fault centroids via area-weighted SVD.

    Returns (origin, strike_hat, dip_hat, normal_hat) in TM-metre coords.

    Conventions (SeisSol: z positive upward):
      normal_hat  – points out of the fault, upward if possible
      strike_hat  – horizontal (z≈0), perpendicular to the dip direction
      dip_hat     – points down-dip (z-component < 0, i.e. deeper)
    """
    weights = areas / areas.sum()
    origin = np.average(centroids, axis=0, weights=weights)
    rel = centroids - origin
    cov = (rel * weights[:, None]).T @ rel
    _, _, Vt = np.linalg.svd(cov)
    # Row 2 of Vt = direction of minimum variance = fault normal
    normal = Vt[2].copy()
    if normal[2] < 0:           # ensure upward component
        normal = -normal

    # Strike: horizontal direction perpendicular to normal
    up = np.array([0.0, 0.0, 1.0])
    strike = np.cross(normal, up)
    norm_s = np.linalg.norm(strike)
    if norm_s < 1e-9:           # near-horizontal fault: use x-axis
        strike = np.array([1.0, 0.0, 0.0])
    else:
        strike /= norm_s

    # Dip: perpendicular to strike and normal, pointing down-dip
    dip = np.cross(strike, normal)
    dip /= np.linalg.norm(dip)
    if dip[2] > 0:              # ensure downward (deeper) direction
        dip = -dip
        strike = -strike        # keep right-hand rule: strike × dip = normal

    return origin, strike, dip, normal


def build_rect_grid(centroids, areas, Sls_c, Sld_c, slip_c, trup_c, trise_c,
                    mu_c, origin, strike_hat, dip_hat, cell_size_m, fwd, hx, hy):
    """Bin triangle centroids onto a regular rectangular fault grid.

    Cell position: area-weighted centroid of binned triangles mapped back to 3-D.
    Slip vector  : area-weighted average of (Sls, Sld); rake recomputed from it.
    sf_moment    : exact sum of mu*slip*area contributions from member triangles.

    Returns a DataFrame with the standard USGS FFM columns plus
    along_strike_km and along_dip_km (position in fault-plane coordinates).
    """
    rel = centroids - origin
    s = rel @ strike_hat    # along-strike coord (m)
    d = rel @ dip_hat       # along-dip coord (m, positive = deeper)

    cs = cell_size_m
    s_edges = np.arange(s.min() - cs / 2, s.max() + cs, cs)
    d_edges = np.arange(d.min() - cs / 2, d.max() + cs, cs)
    ns, nd = len(s_edges) - 1, len(d_edges) - 1

    si = np.clip(np.searchsorted(s_edges, s, side="right") - 1, 0, ns - 1)
    di = np.clip(np.searchsorted(d_edges, d, side="right") - 1, 0, nd - 1)
    cell_idx = si * nd + di
    n_cells = ns * nd

    # Accumulate area-weighted quantities
    sum_a        = np.zeros(n_cells)
    sum_Sls      = np.zeros(n_cells)
    sum_Sld      = np.zeros(n_cells)
    sum_trup     = np.zeros(n_cells)
    sum_trise    = np.zeros(n_cells)
    sum_M0       = np.zeros(n_cells)   # exact M0 per cell
    # Weighted centroid in fault-plane coords (for accurate back-projection)
    sum_s        = np.zeros(n_cells)
    sum_d        = np.zeros(n_cells)
    sum_z        = np.zeros(n_cells)   # for depth (area-weighted)

    np.add.at(sum_a,     cell_idx, areas)
    np.add.at(sum_Sls,   cell_idx, Sls_c * areas)
    np.add.at(sum_Sld,   cell_idx, Sld_c * areas)
    np.add.at(sum_trup,  cell_idx, trup_c * areas)
    np.add.at(sum_trise, cell_idx, trise_c * areas)
    np.add.at(sum_M0,    cell_idx, mu_c * slip_c * areas)
    np.add.at(sum_s,     cell_idx, s * areas)
    np.add.at(sum_d,     cell_idx, d * areas)
    np.add.at(sum_z,     cell_idx, centroids[:, 2] * areas)

    rows = []
    for k in np.where(sum_a > 0)[0]:
        k_s = k // nd
        k_d = k % nd
        a_w = sum_a[k]

        # Area-weighted centroid in fault-plane coords
        sc = sum_s[k] / a_w
        dc = sum_d[k] / a_w
        z_c = sum_z[k] / a_w           # SeisSol z (negative = depth)

        # Back-project centroid to TM metres
        center_tm = origin + sc * strike_hat + dc * dip_hat
        center_tm[2] = z_c             # use area-weighted z for accuracy
        lon_c, lat_c = fwd.transform(center_tm[0], center_tm[1])

        Sls_w = sum_Sls[k] / a_w
        Sld_w = sum_Sld[k] / a_w
        slip_w = float(np.sqrt(Sls_w ** 2 + Sld_w ** 2))
        rake_w = float(np.degrees(np.arctan2(-Sld_w, Sls_w)))

        rows.append({
            "lat":            round(float(lat_c), 4),
            "lon":            round(float(lon_c), 4),
            "x_ew":           round(float((center_tm[0] - hx) / 1e3), 4),
            "y_ns":           round(float((center_tm[1] - hy) / 1e3), 4),
            "z_depth":        round(float(-center_tm[2] / 1e3), 4),
            "along_strike_km": round(float(sc / 1e3), 3),
            "along_dip_km":   round(float(dc / 1e3), 3),
            "cell_strike_km": round(float(0.5 * (s_edges[k_s] + s_edges[k_s+1]) / 1e3), 3),
            "cell_dip_km":    round(float(0.5 * (d_edges[k_d] + d_edges[k_d+1]) / 1e3), 3),
            "slip":           round(slip_w, 4),
            "rake":           round(rake_w, 4),
            "trup":           round(float(sum_trup[k] / a_w), 2),
            "trise":          round(float(sum_trise[k] / a_w), 2),
            "sf_moment":      f"{sum_M0[k]:.3g}",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Refine SeisSol fault triangles and/or rebin to rectangular grid.")
    ap.add_argument("xdmf",    help="SeisSol fault XDMF file")
    ap.add_argument("yaml",    help="Scenario fault YAML (for hypocenter)")
    ap.add_argument("out_csv", help="Output triangular USGS FFM CSV path")
    ap.add_argument("--rise", choices=["ide", "threshold", "p5_p95"],
                    default="ide", help="Rise-time definition (default: ide)")
    ap.add_argument("--levels", type=int, default=1,
                    help="Midpoint-subdivision levels (0=no refinement; "
                         "N multiplies count by 4**N). Default 1.")
    ap.add_argument("--rect-csv",
                    help="If given, also write a rectangular-grid CSV here.")
    ap.add_argument("--rect-cell-size", type=float, default=10.0,
                    metavar="KM",
                    help="Rectangular cell size in km (default 10.0).")
    args = ap.parse_args()

    if args.levels < 0:
        raise SystemExit("--levels must be >= 0")
    if args.rect_csv and args.rect_cell_size <= 0:
        raise SystemExit("--rect-cell-size must be > 0")

    # --- Hypocenter ---
    hx, hy, hz = parse_hypocenter(args.yaml)
    fwd = Transformer.from_crs(SEISSOL_PROJ, "EPSG:4326", always_xy=True)
    hlon, hlat = fwd.transform(hx, hy)
    print(f"Hypocenter (TM m) : x={hx:.1f}, y={hy:.1f}, z={hz:.1f}")
    print(f"Hypocenter (WGS84): lon={hlon:.4f}, lat={hlat:.4f}, "
          f"depth={-hz/1000:.3f} km")

    # --- Read SeisSol ---
    print(f"\nReading {args.xdmf} ...")
    sx = seissolxdmf.seissolxdmf(args.xdmf)
    geom = sx.ReadGeometry()
    conn = sx.ReadConnect()
    ndt  = sx.ReadNdt()
    times = np.asarray(sx.ReadTimes())
    print(f"  Nodes: {geom.shape[0]}, Elements: {conn.shape[0]}, Steps: {ndt}")

    Sls_p = sx.ReadData("Sls", ndt - 1)
    Sld_p = sx.ReadData("Sld", ndt - 1)
    slip_p = np.sqrt(Sls_p ** 2 + Sld_p ** 2)
    RT_p   = sx.ReadData("RT", ndt - 1)
    rake_p = np.degrees(np.arctan2(-Sld_p, Sls_p))

    # --- Parent triangle geometry ---
    pv0 = geom[conn[:, 0]]
    pv1 = geom[conn[:, 1]]
    pv2 = geom[conn[:, 2]]
    parent_centroid = (pv0 + pv1 + pv2) / 3.0
    parent_area = 0.5 * np.linalg.norm(np.cross(pv1 - pv0, pv2 - pv0), axis=1)
    mu_p = mu_avg(parent_centroid[:, 2])

    M0_orig = float(np.sum(mu_p * slip_p * parent_area))

    print("\nGrid size — original triangles:")
    print_grid_stats(parent_area, "original")

    # --- Rise time on parent triangles ---
    print(f"\nReading SRs/SRd time series ({args.rise} method) ...")
    SR_mag = np.empty((ndt, conn.shape[0]), dtype=np.float64)
    for k in range(ndt):
        srs = sx.ReadData("SRs", k)
        srd = sx.ReadData("SRd", k)
        SR_mag[k] = np.sqrt(srs ** 2 + srd ** 2)
    trise_p, slipped_p = compute_rise_time(SR_mag, times, slip_p, args.rise)

    # --- Subdivide ---
    n_orig = conn.shape[0]
    n_fine = n_orig * (4 ** args.levels)
    print(f"\nSubdividing: levels={args.levels}, "
          f"{n_orig} → {n_fine} triangles ...")
    cv0, cv1, cv2, parent = subdivide_triangles(pv0, pv1, pv2, args.levels)
    centroid = (cv0 + cv1 + cv2) / 3.0
    area     = 0.5 * np.linalg.norm(np.cross(cv1 - cv0, cv2 - cv0), axis=1)

    # Inherit parent quantities
    slip  = slip_p[parent]
    Sls   = Sls_p[parent]
    Sld   = Sld_p[parent]
    rake  = rake_p[parent]
    RT    = RT_p[parent]
    trise = trise_p[parent]
    mu    = mu_p[parent]
    sf_moment = mu * slip * area

    if args.levels > 0:
        print("\nGrid size — refined triangles:")
        print_grid_stats(area, f"levels={args.levels}")

    # --- Projected coords ---
    lon_t, lat_t = fwd.transform(centroid[:, 0], centroid[:, 1])
    x_ew  = (centroid[:, 0] - hx) / 1e3
    y_ns  = (centroid[:, 1] - hy) / 1e3
    z_dep = -centroid[:, 2] / 1e3

    # --- Write triangular CSV ---
    df_tri = pd.DataFrame({
        "lat":       np.round(lat_t, 4),
        "lon":       np.round(lon_t, 4),
        "x_ew":      np.round(x_ew, 4),
        "y_ns":      np.round(y_ns, 4),
        "z_depth":   np.round(z_dep, 4),
        "slip":      np.round(slip, 4),
        "rake":      np.round(rake, 4),
        "trup":      np.round(RT, 2),
        "trise":     np.round(trise, 2),
        "sf_moment": [f"{v:.3g}" for v in sf_moment],
    })
    df_tri.to_csv(args.out_csv, index=False)

    M0_tri = float(np.sum(sf_moment))
    print(f"\nWrote triangular CSV: {args.out_csv}  ({len(df_tri)} subfaults)")
    print(f"  Total M0    : orig={M0_orig:.6e}  tri={M0_tri:.6e}  "
          f"rel.diff={(M0_tri - M0_orig) / M0_orig:+.2e} N.m")
    print(f"  Mean slip   : {slip.mean():.4f} m  (exact = "
          f"{slip_p.mean():.4f} m original)")
    print(f"  Area-w slip : {np.sum(slip*area)/np.sum(area):.4f} m  (orig "
          f"{np.sum(slip_p*parent_area)/np.sum(parent_area):.4f} m)")
    slipped = slipped_p[parent]
    if slipped.any():
        print(f"  Rise time   : min/mean/max = "
              f"{trise[slipped].min():.2f} / {trise[slipped].mean():.2f} / "
              f"{trise[slipped].max():.2f} s")

    # --- Optional rectangular grid ---
    if args.rect_csv:
        cell_m = args.rect_cell_size * 1e3
        print(f"\nBuilding rectangular grid  (cell size = {args.rect_cell_size} km) ...")
        origin, strike_hat, dip_hat, normal_hat = find_fault_frame(centroid, area)

        strike_az = float(np.degrees(np.arctan2(strike_hat[0], strike_hat[1]))) % 360
        dip_ang = float(np.degrees(np.arccos(np.clip(-dip_hat[2], -1, 1))))
        print(f"  Fault plane: strike≈{strike_az:.1f}°, dip≈{dip_ang:.1f}°")
        print(f"  Normal      : ({normal_hat[0]:.3f}, {normal_hat[1]:.3f}, "
              f"{normal_hat[2]:.3f})")

        df_rect = build_rect_grid(
            centroid, area, Sls, Sld, slip, RT, trise, mu,
            origin, strike_hat, dip_hat, cell_m, fwd, hx, hy)

        df_rect.to_csv(args.rect_csv, index=False)

        M0_rect = sum(float(v) for v in df_rect["sf_moment"])
        rect_area_km2 = (args.rect_cell_size ** 2) * len(df_rect)
        print(f"  Wrote rectangular CSV: {args.rect_csv}  ({len(df_rect)} cells)")
        print(f"\nGrid size — rectangular cells ({args.rect_cell_size} km):")
        print(f"  [rectangular]  n={len(df_rect)}")
        print(f"    Cell size   : {args.rect_cell_size:.3f} × "
              f"{args.rect_cell_size:.3f} km")
        print(f"    Equiv side  : {args.rect_cell_size:.3f} km (uniform)")
        print(f"    Total area  : {rect_area_km2:.1f} km^2")
        print(f"  Total M0    : orig={M0_orig:.6e}  rect={M0_rect:.6e}  "
              f"rel.diff={(M0_rect - M0_orig) / M0_orig:+.2e} N.m")
        slip_rect = df_rect["slip"].values
        area_rect = np.full(len(df_rect), (args.rect_cell_size * 1e3) ** 2)
        print(f"  Mean slip   : {slip_rect.mean():.4f} m")
        print(f"  Area-w slip : {np.sum(slip_rect * area_rect) / area_rect.sum():.4f} m")


if __name__ == "__main__":
    main()
