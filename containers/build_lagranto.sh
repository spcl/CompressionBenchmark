#!/usr/bin/env bash
# Unmodified public LAGRANTO ECMWF 2.7, using its CDO/CF input reader.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${LAGRANTO_BIN_DIR:-$ROOT/compressor/bin}
WORK=$(mktemp -d)
trap 'rm -rf -- "$WORK"' EXIT
python3 "$ROOT/containers/fetch_lagranto.py" "$WORK"
cd "$WORK"
FC=${FC:-gfortran}
NF_CONFIG=${NF_CONFIG:-nf-config}
read -ra NETCDF_FFLAGS <<< "$("$NF_CONFIG" --fflags)"
read -ra NETCDF_LIBS <<< "$("$NF_CONFIG" --flibs)"
"$FC" -c -O -fallow-argument-mismatch -ffixed-line-length-none "${NETCDF_FFLAGS[@]}" ./*.f
"$FC" -o caltra ./*.o "${NETCDF_LIBS[@]}"
"$FC" --version >> caltra.build-info
"$NF_CONFIG" --version >> caltra.build-info
mkdir -p "$BIN"
install -m 755 caltra "$BIN/caltra"
install -m 644 caltra.build-info "$BIN/caltra.build-info"
