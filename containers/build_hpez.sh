#!/usr/bin/env bash
# Public HPEZ plus bug fixes, with a separate benchmark C ABI wrapper.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${HPEZ_BIN_DIR:-$ROOT/compressor/bin}
REV=291699d06d754f58e074d2b2a3a355f45a239e34
WORK=$(mktemp -d)
trap 'rm -rf -- "$WORK"' EXIT
git clone --no-checkout "${HPEZ_SOURCE_REPO:-https://github.com/JLiu-1/HPEZ-QoZ2.0.git}" "$WORK/source"
git -C "$WORK/source" checkout --detach "$REV"
git -C "$WORK/source" apply --check "$ROOT/compressor/patches/hpez-bugfixes.patch"
git -C "$WORK/source" apply "$ROOT/compressor/patches/hpez-bugfixes.patch"
sed -i 's/add_library(zstd SHARED/add_library(zstd STATIC/' "$WORK/source/zstd/CMakeLists.txt"
cp "$ROOT/compressor/native/hpez.cpp" "$WORK/source/compbench_hpez.cpp"
cat >> "$WORK/source/CMakeLists.txt" <<'CMAKE'
add_library(compbench_hpez SHARED compbench_hpez.cpp)
target_link_libraries(compbench_hpez PRIVATE QoZ)
CMAKE
cmake -S "$WORK/source" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' -DCMAKE_C_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DOMP=OFF -DQoZ_DEBUG_TIMINGS=OFF -DQoZ_USE_BUNDLED_ZSTD=ON
cmake --build "$WORK/build" --target hpez compbench_hpez --parallel "${BUILD_JOBS:-4}"
mkdir -p "$BIN"
install -m 755 "$WORK/build/test/hpez" "$BIN/hpez"
install -m 755 "$WORK/build/libcompbench_hpez.so" "$BIN/libcompbench_hpez.so"
printf 'HPEZ public revision: %s\n' "$REV" > "$BIN/hpez.build-info"
sha256sum "$ROOT/compressor/patches/hpez-bugfixes.patch" >> "$BIN/hpez.build-info"
sha256sum "$ROOT/compressor/native/hpez.cpp" "$BIN/libcompbench_hpez.so" >> "$BIN/hpez.build-info"
