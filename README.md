# Compression benchmarks

`compbench` is a command-line tool for comparing error-bounded compressors on
weather and climate benchmarks. It supports EBCC, SZ, SZ3, HPEZ, SPERR and MGARD.
It runs rate-distortion, throughput and application benchmarks locally or
through Slurm, and generates figures from saved measurements.

## Benchmarks and dataset sizes

Sizes below are approximate uncompressed float32 inputs for the default
[configuration](configs/benchmark.yaml), using decimal MB/GB/TB. Downloads,
NetCDF metadata, scratch files and results require additional disk space.

| Benchmark | Purpose | Sample tier | Full tier |
| --- | --- | --- | --- |
| `rate_distortion` | Compression ratio, PSNR and SSIM | 0.94 GB (one timestep) | 8.22 TB (hourly, all of 2023) |
| `kappa_study` | Effect of EBCC's κ parameter | 0.94 GB | 0.94 GB |
| `quantization_residual` | Base quantization and residual variants | 0.94 GB | 0.94 GB |
| `error_statistics` | Error distributions and spectra | 0.94 GB | 0.94 GB |
| `residual_layer_ablation` | Synthetic Gaussian residual experiment | 0.26 MB (generated) | 0.26 MB (generated) |
| `case_study_hurricane` | Hurricane Iota wind and divergence | 16.6 MB | 16.6 MB |
| `sequential_throughput` | Sequential compression/decompression throughput | 0.94 GB | 0.94 GB |
| `weak_scaling` | Parallel throughput | 3.75 GB (4 timesteps) | 961.1 GB (1,024 timesteps) |
| `energy_budget_closure` | Energy budget closure analysis; only member 0 is compressed | 3.84 GB/member; 7.69 GB total (2 members, 6 days) | 117.84 GB/member; 1.18 TB total (10 members, August–October 2016) |
| `trajectory_simulation` | LAGRANTO trajectories and particle density; only member 0 is compressed | 1.05 GB/member; 2.10 GB total (2 members, 24 hours) | 5.71 GB/member; 57.06 GB total (10 members, 6 days) |

The κ, quantization, error-statistics and sequential-throughput benchmarks reuse the
March 1, 2023 timestep from `rate_distortion` in both tiers; their sizes refer
to that shared prepared dataset, even when a benchmark uses only selected fields.
Energy inputs are six-hourly for sample and three-hourly for full. Trajectory
inputs are three-hourly and include the final interpolation endpoint; sample
runs use every 100th starting particle, while full runs use all 35,298 particles.
For both ensemble benchmarks, total sizes include all members; members other than 0
remain uncompressed and provide the ensemble uncertainty baseline.

## Install

Run these commands from the repository root on Linux with Python 3.11+, `uv`,
Git, CMake and a C/C++ compiler installed. Native dependencies are also provided
by the [container recipe](containers/Containerfile).

```bash
uv sync --locked
bash containers/build_hpez.sh
bash compressor/build_mgard_x.sh
export HPEZ_LIBRARY_PATH="$PWD/compressor/bin/libcompbench_hpez.so"
export MGARD_LIBRARY_PATH="$PWD/compressor/bin/libcompbench_mgard.so"
uv run --locked compbench doctor
```

`uv sync --locked` installs the pinned Python dependencies into `.venv`; the build
scripts compile HPEZ and MGARD, and the environment variables tell `compbench`
where to find their libraries. `uv run --locked` runs commands in that environment.
`doctor` checks whether the configured compressors can be loaded and reports
whether the LAGRANTO executable (`caltra`) is available.

Trajectory runs additionally require LAGRANTO: install `gfortran` and NetCDF
Fortran (`nf-config`), run `bash containers/build_lagranto.sh`, and add
`$PWD/compressor/bin` to `PATH`.

## Run a benchmark

Start with the sample rate-distortion benchmark (about 0.94 GB of input):

```bash
uv run --locked compbench datasets fetch rate_distortion --tier sample
uv run --locked compbench plan rate_distortion --tier sample --profile configs/local.yaml
uv run --locked compbench run rate_distortion --tier sample --profile configs/local.yaml
uv run --locked compbench plot rate_distortion --tier sample --profile configs/local.yaml
```

| Command | Purpose |
| --- | --- |
| `datasets fetch NAME` | Download and prepare inputs for the named benchmark. |
| `plan` | Preview the workload, settings and output location without running it. |
| `run` | Run the benchmark and save measurements, resuming matching completed tasks. |
| `plot` | Validate saved measurements and generate figures from them. |

For `plan`, `run`, `plot`, `validate` and `submit`, put the benchmark name directly
after the command (for example, `compbench run rate_distortion`). Options control
the dataset tier and execution settings:

- `--tier sample`: use the smaller demonstration dataset (the default).
  `--tier full` uses the complete configured dataset; rate-distortion inputs
  total about 8.22 TB. Use the same tier for fetching, running and plotting.
- `--profile FILE`: load execution settings such as paths, worker count and Slurm
  resources. Edit [configs/local.yaml](configs/local.yaml) for local runs.
- `--config FILE`: load scientific settings such as compressors and error targets.
  Defaults to [configs/benchmark.yaml](configs/benchmark.yaml).

With the local profile, inputs go to `data/prepared/` and results to
`results/sample/rate_distortion/`. Paths resolve from the directory where
you invoke the command.

Use `compbench run --help` (prefixed with `uv run --locked` as above) to show options and list
workflows, including `kappa_study`, `error_statistics`, `sequential_throughput`,
`weak_scaling`, `energy_budget_closure` and `trajectory_simulation`. Fetch the matching
dataset before running; `residual_layer_ablation` uses synthetic data and needs
no download. Energy and trajectory downloads require CDS credentials and
accepted dataset terms.

`reproduce` runs all workflows to reproduce the figures in the EBCC paper. Use
`--tier full` for the paper's full datasets; `--tier sample` generates demonstration
figures from smaller datasets. Prepare the datasets first:

```bash
uv run --locked compbench reproduce --tier sample --profile configs/local.yaml
```

Use `--workflows NAME ...` to select a subset. The current cluster configuration
supports a container environment with Slurm on the Alps cluster and requires
adjustments for other cluster environments. For cluster runs, edit
[configs/slurm.yaml](configs/slurm.yaml) and preview a submission:

```bash
uv run --locked compbench submit rate_distortion --tier full --profile configs/slurm.yaml --dry-run
```

`submit` creates Slurm job scripts and submits them; `--dry-run` only generates
the scripts for inspection. Remove `--dry-run` to submit. After the jobs finish,
use `validate` to check result completeness, configuration consistency and error
bounds, then `plot` to generate figures. Use the same workflow, tier and profile.
