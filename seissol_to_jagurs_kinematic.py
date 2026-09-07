"""
Convert SeisSol free-surface XDMF to a time-varying kinematic displacement
input for JAGURS: one GMT-compatible NETCDF3_CLASSIC GRD file per time step,
plus an index file listing (step, time, filename).

GRD format matches seissol_to_jagurs_grd.py (x_range, y_range, z_range,
spacing, dimension, z), so every step file is a drop-in replacement for
the static JAGURS seafloor-displacement input.

Usage:
    python seissol_to_jagurs_kinematic.py <surface.xdmf> <out_dir>
           [--prefix NAME]
           [--spacing DEG]
           [--bbox LON_MIN LON_MAX LAT_MIN LAT_MAX]
           [--subsample N]
           [--xyz-threshold M] [--xyz-all]

Output (in <out_dir>):
    <prefix>_0000.grd, <prefix>_0001.grd, ...
    <prefix>_index.txt       (columns: step  time_s  filename)
    <prefix>.xyz             (columns: time lon lat disp; rows with
                              |disp| below threshold are skipped by default)
"""

import argparse
import os
import numpy as np
import netCDF4 as nc
import seissolxdmf
from pyproj import Transformer
from scipy.spatial import Delaunay

SEISSOL_PROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lat_0=0 +lon_0=100"


def build_grid(lon_n, lat_n, v0, v1, v2, spacing, bbox):
    if spacing is None:
        edges = np.concatenate([
            np.linalg.norm(v1 - v0, axis=1),
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v0 - v2, axis=1),
        ])
        mean_edge_m = float(edges.mean())
        lat_c = 0.5 * (lat_n.min() + lat_n.max())
        m_per_deg = 111320.0 * 0.5 * (1.0 + np.cos(np.radians(lat_c)))
        spacing = round(mean_edge_m / m_per_deg, 3)

    if bbox is None:
        lon_min = float(np.floor(lon_n.min() / spacing) * spacing)
        lon_max = float(np.ceil(lon_n.max() / spacing) * spacing)
        lat_min = float(np.floor(lat_n.min() / spacing) * spacing)
        lat_max = float(np.ceil(lat_n.max() / spacing) * spacing)
    else:
        lon_min, lon_max, lat_min, lat_max = bbox

    nx = int(np.floor((lon_max - lon_min) / spacing)) + 1
    ny = int(np.floor((lat_max - lat_min) / spacing)) + 1
    grid_lon = lon_min + np.arange(nx) * spacing
    grid_lat = lat_min + np.arange(ny) * spacing
    GLO, GLA = np.meshgrid(grid_lon, grid_lat)
    grid_pts = np.column_stack([GLO.ravel(), GLA.ravel()])
    return spacing, grid_lon, grid_lat, nx, ny, grid_pts


def precompute_interp(node_ll, node_mask, grid_pts):
    """Delaunay + barycentric weights — built once, reused for every time step."""
    valid_idx = np.where(node_mask)[0]
    tri = Delaunay(node_ll[valid_idx])
    simplex = tri.find_simplex(grid_pts)
    inside = simplex >= 0

    X = tri.transform[simplex[inside]]                # (N_in, 3, 2)
    rel = grid_pts[inside] - X[:, 2]                  # (N_in, 2)
    b = np.einsum("ijk,ik->ij", X[:, :2], rel)        # (N_in, 2)
    w = np.zeros((len(grid_pts), 3))
    w[inside, :2] = b
    w[inside, 2] = 1.0 - b.sum(axis=1)

    verts = np.zeros((len(grid_pts), 3), dtype=np.int64)
    verts[inside] = valid_idx[tri.simplices[simplex[inside]]]
    return inside, verts, w


def write_grd(path, grid_lon, grid_lat, z_grid, spacing, source):
    """Write a NETCDF3_CLASSIC GRD in the same layout as seissol_to_jagurs_grd.py."""
    nx, ny = len(grid_lon), len(grid_lat)
    # GMT convention: row 0 = max latitude
    z_out = np.flipud(z_grid).astype(np.float32).ravel()

    ds = nc.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("side", 2)
    ds.createDimension("xysize", nx * ny)

    x_var = ds.createVariable("x_range", "f8", ("side",))
    x_var[:] = [grid_lon[0], grid_lon[-1]]
    x_var.units = "degrees"

    y_var = ds.createVariable("y_range", "f8", ("side",))
    y_var[:] = [grid_lat[0], grid_lat[-1]]
    y_var.units = "degrees"

    z_var = ds.createVariable("z_range", "f8", ("side",))
    z_var[:] = [float(z_out.min()), float(z_out.max())]
    z_var.units = "meters"

    sp_var = ds.createVariable("spacing", "f8", ("side",))
    sp_var[:] = [spacing, spacing]

    dim_var = ds.createVariable("dimension", "i4", ("side",))
    dim_var[:] = [nx, ny]

    z_data = ds.createVariable("z", "f4", ("xysize",))
    z_data[:] = z_out
    z_data.scale_factor = np.float64(1.0)
    z_data.add_offset = np.float64(0.0)
    z_data.node_offset = np.int32(0)

    ds.title = "SeisSol kinematic seafloor displacement for JAGURS"
    ds.source = source
    ds.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xdmf", help="SeisSol free-surface XDMF file")
    ap.add_argument("out_dir", help="Output directory for .grd files")
    ap.add_argument("--prefix", default="disp",
                    help="File prefix (default 'disp' -> disp_0000.grd ...)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="Grid spacing in degrees (default: from mean mesh edge)")
    ap.add_argument("--bbox", type=float, nargs=4,
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    default=None,
                    help="Grid extent in degrees (default: tight around mesh)")
    ap.add_argument("--subsample", type=int, default=1,
                    help="Keep every Nth time step (default 1)")
    ap.add_argument("--xyz-threshold", type=float, default=1e-6,
                    help="XYZ: skip rows with |disp| below this (default 1e-6 m)")
    ap.add_argument("--xyz-all", action="store_true",
                    help="XYZ: write every grid point including zeros")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Read SeisSol ---
    print(f"Reading {args.xdmf} ...")
    sx = seissolxdmf.seissolxdmf(args.xdmf)
    geom = sx.ReadGeometry()
    conn = sx.ReadConnect()
    ndt = sx.ReadNdt()
    times = np.asarray(sx.ReadTimes())
    print(f"  Nodes: {geom.shape[0]}, Elements: {conn.shape[0]}, Steps: {ndt}")
    print(f"  Time range: {times[0]:.2f} .. {times[-1]:.2f} s")

    # --- Node positions in lon/lat (once) ---
    fwd = Transformer.from_crs(SEISSOL_PROJ, "EPSG:4326", always_xy=True)
    lon_n, lat_n = fwd.transform(geom[:, 0], geom[:, 1])
    node_ll = np.column_stack([lon_n, lat_n])

    # --- Triangle areas for cell-center -> node-center averaging ---
    v0, v1, v2 = geom[conn[:, 0]], geom[conn[:, 1]], geom[conn[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    node_weights = np.zeros(geom.shape[0], dtype=np.float64)
    for i in range(3):
        np.add.at(node_weights, conn[:, i], areas)
    node_mask = node_weights > 0

    # --- Grid ---
    spacing, grid_lon, grid_lat, nx, ny, grid_pts = build_grid(
        lon_n, lat_n, v0, v1, v2, args.spacing, args.bbox)
    print(f"  Grid spacing: {spacing} deg")
    print(f"  Grid bbox   : lon [{grid_lon[0]:.3f}, {grid_lon[-1]:.3f}], "
          f"lat [{grid_lat[0]:.3f}, {grid_lat[-1]:.3f}]")
    print(f"  Grid size   : {nx} x {ny} = {nx*ny} points")

    # --- Precompute triangulation + bary weights (once) ---
    print("  Precomputing Delaunay triangulation ...")
    inside, verts, w = precompute_interp(node_ll, node_mask, grid_pts)
    print(f"  Points inside mesh: {int(inside.sum())}/{len(grid_pts)}")

    # --- Stream through time steps ---
    steps = list(range(0, ndt, max(args.subsample, 1)))
    total = len(steps)
    index_path = os.path.join(args.out_dir, f"{args.prefix}_index.txt")
    xyz_path = os.path.join(args.out_dir, f"{args.prefix}.xyz")
    print(f"  Writing {total} .grd files + {xyz_path} to {args.out_dir}/ ...")

    n_nodes = geom.shape[0]
    src_tag = os.path.basename(args.xdmf)

    with open(index_path, "w") as idx, open(xyz_path, "w") as xyz:
        idx.write("# step  time_s  filename\n")
        xyz.write("# time  lon  lat  disp\n")
        for ki, k in enumerate(steps):
            u_cell = sx.ReadData("u3", k)

            # Cell-centered -> node-centered (area-weighted)
            node_vals = np.zeros(n_nodes, dtype=np.float64)
            for i in range(3):
                np.add.at(node_vals, conn[:, i], u_cell * areas)
            node_vals[node_mask] /= node_weights[node_mask]

            # Interpolate to grid via precomputed bary weights
            grid_vals = np.zeros(len(grid_pts))
            grid_vals[inside] = (node_vals[verts[inside]] * w[inside]).sum(axis=1)
            z_grid = grid_vals.reshape(ny, nx)

            # --- GRD ---
            fname = f"{args.prefix}_{ki:04d}.grd"
            fpath = os.path.join(args.out_dir, fname)
            write_grd(fpath, grid_lon, grid_lat, z_grid, spacing,
                      source=f"seissol_to_jagurs_kinematic.py step {k} "
                             f"(t={times[k]:.3f}s) from {src_tag}")
            idx.write(f"{ki:6d}  {float(times[k]):10.4f}  {fname}\n")

            # --- XYZ (streaming) ---
            t = float(times[k])
            if args.xyz_all:
                keep = np.arange(len(grid_pts))
            else:
                keep = np.where(np.abs(grid_vals) > args.xyz_threshold)[0]
            if len(keep):
                rows = np.column_stack([
                    np.full(len(keep), t),
                    grid_pts[keep, 0],
                    grid_pts[keep, 1],
                    grid_vals[keep],
                ])
                np.savetxt(xyz, rows, fmt="%.4f %.6f %.6f %.6f")

            if total > 0 and (ki + 1) % max(total // 20, 1) == 0:
                print(f"    step {ki+1}/{total}  t={times[k]:.2f}s  "
                      f"z range [{z_grid.min():.4f}, {z_grid.max():.4f}] m  "
                      f"xyz_rows={len(keep)}")

    print(f"Index: {index_path}")
    print(f"XYZ  : {xyz_path}")
    print("Done.")


if __name__ == "__main__":
    main()
