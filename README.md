# SeisSol_PostProceesing_Scripts

Post-processing scripts for SeisSol dynamic-rupture fault outputs: computing
seismic moment/magnitude, converting to USGS finite-fault-model (FFM) format,
converting seafloor displacement to JAGURS tsunami-model input, and plotting.

All scripts read SeisSol XDMF fault/surface output (via `seissolxdmf` or
`h5py`) and share a common 1-D depth-dependent rigidity (mu) profile used to
compute seismic moment.

## Scripts

### `computeMw.py`
Computes moment magnitude (Mw), total seismic moment (M0), total fault area,
slipped area, and average slip directly from an `.h5` fault output file
(reads `geometry`/`connect`/`SRs`/`ASl` or `Sls`/`Sld` datasets via `h5py`).

```
python computeMw.py file.h5
```

### `computeMw_bin.py`
Same computation as `computeMw.py`, but uses `seissolxdmf` so it works with
both `.h5` and `.bin` XDMF backends. Vectorized with numpy and accepts a
configurable slip threshold.

```
python computeMw_bin.py <fault.xdmf> [--slip-threshold 0.03]
```

### `seissol_to_usgs_ffm.py`
Converts a SeisSol fault XDMF output into a USGS finite-fault-model CSV, one
row per fault triangle: centroid lat/lon/depth, position relative to the
hypocenter, slip, rake, rupture time, rise time, and subfault moment. The
hypocenter is parsed from the scenario's `forced_rupture_time` YAML block.

```
python seissol_to_usgs_ffm.py <fault.xdmf> <scenario_fault.yaml> <out.csv> [--rise ide|threshold|p5_p95]
```

### `seissol_to_usgs_ffm_refined.py`
Extended version of `seissol_to_usgs_ffm.py` with two extra stages:
1. **Midpoint subdivision** (`--levels N`): splits each fault triangle into
   `4**N` equal-area sub-triangles, preserving total M0 and mean slip exactly.
2. **Rectangular rebinning** (`--rect-csv`, `--rect-cell-size`): projects
   sub-triangle centroids onto the best-fit fault plane and bins them onto a
   regular along-strike x along-dip grid, producing a second CSV with
   uniform rectangular subfaults.

```
python seissol_to_usgs_ffm_refined.py <fault.xdmf> <scenario.yaml> <out_tri.csv> \
       [--rise ide|threshold|p5_p95] [--levels N] \
       [--rect-csv out_rect.csv] [--rect-cell-size KM]
```

### `plot_output_from_seissol_to_usgs_ffm.py`
Plots the slip distribution from a USGS-FFM-style CSV (e.g. output of the two
scripts above) as a coastline map colored by slip magnitude, using
matplotlib/cartopy.

```
python plot_output_from_seissol_to_usgs_ffm.py --input_file nuc3output.csv
```

### `seissol_to_jagurs_grd.py`
Converts a static SeisSol free-surface vertical displacement (`u3`) XDMF
output into a JAGURS-compatible GMT GRD file (NETCDF3_CLASSIC): reprojects
from the SeisSol Transverse Mercator grid to WGS84, interpolates the
unstructured mesh onto a regular structured grid, and writes both `.grd` and
`.xyz` outputs. Configuration (input file, output paths, grid extent) is
hardcoded at the top of the script.

```
python seissol_to_jagurs_grd.py
```

### `seissol_to_jagurs_kinematic.py`
Time-varying version of `seissol_to_jagurs_grd.py`: produces one GRD file per
time step (same format, so each is a drop-in replacement for the static
JAGURS input), plus an index file listing step/time/filename and a combined
XYZ time series. Uses a precomputed Delaunay triangulation and barycentric
weights so the interpolation is only set up once and reused across all time
steps.

```
python seissol_to_jagurs_kinematic.py <surface.xdmf> <out_dir> \
       [--prefix NAME] [--spacing DEG] \
       [--bbox LON_MIN LON_MAX LAT_MIN LAT_MAX] \
       [--subsample N] [--xyz-threshold M] [--xyz-all]
```

## Dependencies

`numpy`, `pandas`, `h5py`, `seissolxdmf`, `pyproj`, `scipy`, `netCDF4`,
`matplotlib`, `cartopy`
