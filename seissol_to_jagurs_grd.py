"""
Convert SeisSol XDMF surface displacement (u3) to JAGURS-compatible GMT GRD file.

Reads unstructured triangular mesh from SeisSol, reprojects from Transverse Mercator
to WGS84 geographic coordinates, interpolates onto a structured regular grid, and
writes a NETCDF3_CLASSIC GRD file.

Grid extent matches DEM SD00 domain. Grid spacing is derived from the SeisSol mesh.
"""

import numpy as np
import seissolxdmf
from pyproj import Transformer
from scipy.interpolate import griddata
import netCDF4 as nc

# --- Configuration ---
XDMF_FILE = "../Mentawai_KEN/mentawai_dynamics_nuc12_nuc3_t500_0.96_surface-surface.xdmf"
OUTPUT_GRD = "disp_seissol_SD00.grd"
OUTPUT_XYZ = "disp_seissol_SD00.xyz"

# SeisSol projection
SEISSOL_PROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lat_0=0 +lon_0=100"

# DEM SD00 extent (hardcoded)
LON_MIN, LON_MAX = 83.85, 120.381
LAT_MIN, LAT_MAX = -17.75, 16.351

# --- Step 1: Read SeisSol XDMF data ---
print("Reading SeisSol XDMF...")
sx = seissolxdmf.seissolxdmf(XDMF_FILE)
geometry = sx.ReadGeometry()       # (nNodes, 3) - XYZ in projected coords
connect = sx.ReadConnect()         # (nElements, 3) - triangle connectivity
u3_cell = sx.ReadData("u3", 0)    # (nElements,) - vertical displacement (cell-centered)

print(f"  Nodes: {geometry.shape[0]}, Elements: {connect.shape[0]}")
print(f"  u3 range: [{u3_cell.min():.4f}, {u3_cell.max():.4f}] m")

# --- Step 2: Convert cell-centered data to node-centered ---
print("Converting cell-centered to node-centered values...")
nNodes = geometry.shape[0]
node_values = np.zeros(nNodes, dtype=np.float64)
node_weights = np.zeros(nNodes, dtype=np.float64)

v0 = geometry[connect[:, 0]]
v1 = geometry[connect[:, 1]]
v2 = geometry[connect[:, 2]]
cross = np.cross(v1 - v0, v2 - v0)
areas = 0.5 * np.linalg.norm(cross, axis=1)

for i in range(3):
    np.add.at(node_values, connect[:, i], u3_cell * areas)
    np.add.at(node_weights, connect[:, i], areas)

mask = node_weights > 0
node_values[mask] /= node_weights[mask]

print(f"  Node u3 range: [{node_values[mask].min():.4f}, {node_values[mask].max():.4f}] m")

# --- Step 3: Reproject from TM to WGS84 lon/lat ---
print("Reprojecting to WGS84 lon/lat...")
transformer = Transformer.from_crs(SEISSOL_PROJ, "EPSG:4326", always_xy=True)
lon, lat = transformer.transform(geometry[:, 0], geometry[:, 1])

print(f"  Lon range: [{lon.min():.4f}, {lon.max():.4f}]")
print(f"  Lat range: [{lat.min():.4f}, {lat.max():.4f}]")

# --- Step 4: Compute structured grid spacing from SeisSol mesh ---
print("Computing grid spacing from mesh resolution...")
edge1 = np.linalg.norm(v1 - v0, axis=1)
edge2 = np.linalg.norm(v2 - v1, axis=1)
edge3 = np.linalg.norm(v0 - v2, axis=1)
mean_edge_m = np.mean(np.concatenate([edge1, edge2, edge3]))

# Convert mean edge length (meters) to degrees (approximate at domain center)
center_lat = (LAT_MIN + LAT_MAX) / 2.0
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
spacing_deg = mean_edge_m / ((m_per_deg_lat + m_per_deg_lon) / 2.0)
# Round to a clean value
spacing_deg = round(spacing_deg, 3)

print(f"  Mean edge length: {mean_edge_m:.1f} m")
print(f"  Grid spacing: {spacing_deg} deg")

# --- Step 5: Create regular grid and interpolate ---
print("Creating structured grid and interpolating...")
nx = int(np.floor((LON_MAX - LON_MIN) / spacing_deg)) + 1
ny = int(np.floor((LAT_MAX - LAT_MIN) / spacing_deg)) + 1

# Build grid from exact spacing so (x_max - x_min) == (nx-1) * spacing
grid_lon = LON_MIN + np.arange(nx) * spacing_deg
grid_lat = LAT_MIN + np.arange(ny) * spacing_deg
grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

print(f"  Grid size: {nx} x {ny} = {nx * ny} points")

valid = mask
points = np.column_stack([lon[valid], lat[valid]])
values = node_values[valid]

z_grid = griddata(points, values, (grid_lon_2d, grid_lat_2d),
                  method="linear", fill_value=0.0)

print(f"  Interpolated z range: [{z_grid.min():.4f}, {z_grid.max():.4f}] m")
print(f"  Non-zero points: {np.count_nonzero(z_grid)}/{z_grid.size}")

# --- Step 6: Write GMT GRD (NETCDF3_CLASSIC) ---
print(f"Writing {OUTPUT_GRD}...")

# GMT GRD convention: row 0 = max latitude (north to south)
z_out = np.flipud(z_grid).astype(np.float32).ravel()

# Recompute exact x/y ranges from the grid
x_range = [grid_lon[0], grid_lon[-1]]
y_range = [grid_lat[0], grid_lat[-1]]

ds = nc.Dataset(OUTPUT_GRD, "w", format="NETCDF3_CLASSIC")
ds.createDimension("side", 2)
ds.createDimension("xysize", nx * ny)

x_var = ds.createVariable("x_range", "f8", ("side",))
x_var[:] = x_range
x_var.units = "degrees"

y_var = ds.createVariable("y_range", "f8", ("side",))
y_var[:] = y_range
y_var.units = "degrees"

z_var = ds.createVariable("z_range", "f8", ("side",))
z_var[:] = [float(z_out.min()), float(z_out.max())]
z_var.units = "meters"

spacing_var = ds.createVariable("spacing", "f8", ("side",))
spacing_var[:] = [spacing_deg, spacing_deg]

dimension_var = ds.createVariable("dimension", "i4", ("side",))
dimension_var[:] = [nx, ny]

z_data = ds.createVariable("z", "f4", ("xysize",))
z_data[:] = z_out
z_data.scale_factor = np.float64(1.0)
z_data.add_offset = np.float64(0.0)
z_data.node_offset = np.int32(0)

ds.title = "SeisSol seafloor displacement for JAGURS"
ds.source = "seissol_to_jagurs_grd.py from mentawai_dynamics_nuc12_nuc3_t500_0.96_surface"

ds.close()
print(f"Done. Output: {OUTPUT_GRD}")

# --- Step 7: Write XYZ file ---
print(f"Writing {OUTPUT_XYZ}...")
with open(OUTPUT_XYZ, "w") as f:
    for j in range(ny):
        for i in range(nx):
            f.write(f"{grid_lon[i]:.6f} {grid_lat[j]:.6f} {z_grid[j, i]:.6f}\n")
print(f"Done. Output: {OUTPUT_XYZ}")
