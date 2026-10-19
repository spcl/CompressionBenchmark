# Required analysis inputs

These inputs are required for energy-closure and trajectory analysis.

- `weights.nc`: the climatological pressure weights used in energy closure.
- `startf`: the 35,298 trajectory starting positions. Sample runs take every 100th row; full runs use every row.

Energy budget closure analysis always reads the installed `weights.nc` directly; no YAML override or prepared-data copy is used. Dataset preparation copies the initial particle file to the configured data directory.
