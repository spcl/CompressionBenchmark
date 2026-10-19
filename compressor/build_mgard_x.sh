#!/usr/bin/env bash
# Build inside the same environment/container used for the benchmarks.
set -euo pipefail
ulimit -c 0
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BIN=${MGARD_BIN_DIR:-$ROOT/bin}
case "${1:-build}" in
    clean)
        rm -f -- "$BIN/libcompbench_mgard.build-info" "$BIN/libcompbench_mgard.so"
        exit 0 ;;
    build) ;;
    *) echo "Usage: $0 [build|clean]" >&2; exit 2 ;;
esac
REV=6b43196d0e300357290c0918bd446143ae4f5ae7
JOBS=${BUILD_JOBS:-8}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/compbench-mgard.XXXXXXXX")
trap 'rm -rf -- "$WORK"' EXIT
mkdir -p "$BIN"
git clone --no-checkout "${MGARD_SOURCE_REPO:-https://github.com/CODARcode/MGARD.git}" "$WORK/source"
git -C "$WORK/source" checkout --detach "$REV"
# Use matching private headers, protoc and library: HPC containers can contain
# incompatible /usr and /usr/local Protobuf installations.
if [[ ${MGARD_SYSTEM_PROTOBUF:-0} != 1 ]]; then
    git clone --depth 1 --branch v3.20.3 https://github.com/protocolbuffers/protobuf.git "$WORK/protobuf"
    cmake -S "$WORK/protobuf/cmake" -B "$WORK/protobuf-build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$WORK/deps" \
        -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    cmake --build "$WORK/protobuf-build" --parallel "$JOBS"
    cmake --install "$WORK/protobuf-build"
    export PATH="$WORK/deps/bin:$PATH"
    export PKG_CONFIG_PATH="$WORK/deps/lib/pkgconfig:$WORK/deps/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
# Link a small C ABI into the same build, preserving pinned compressor defaults.
cp "$ROOT/native/mgard.cpp" "$WORK/source/compbench_mgard.cpp"
cat >> "$WORK/source/CMakeLists.txt" <<'CMAKE'
add_library(compbench_mgard SHARED compbench_mgard.cpp)
target_link_libraries(compbench_mgard PRIVATE mgard-library ${CMAKE_DL_LIBS})
CMAKE
cmake -S "$WORK/source" -B "$WORK/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS:-} ${WORK:+-I$WORK/deps/include}" \
    -DMGARD_ENABLE_SERIAL=ON -DMGARD_ENABLE_OPENMP=OFF -DMGARD_ENABLE_CUDA=OFF \
    -DMGARD_ENABLE_MDR=OFF -DMGARD_ENABLE_CLI=OFF -DBUILD_TESTING=OFF
cmake --build "$WORK/build" --target compbench_mgard --parallel "$JOBS"
LIBRARY=$(find "$WORK/build" -name libcompbench_mgard.so -print -quit)
install -m 755 "$LIBRARY" "$BIN/libcompbench_mgard.so"
{
    echo "MGARD revision: $REV"
    echo "Backend: serial; static MGARD library; Huffman/Zstd available"
    echo "CXXFLAGS: ${CXXFLAGS:-<compiler default CPU target>}"
    grep -E '^CMAKE_(CXX_COMPILER|CXX_FLAGS[^:]*|BUILD_TYPE):' "$WORK/build/CMakeCache.txt"
    uname -sm
    "${CXX:-c++}" --version
    sha256sum "$BIN/libcompbench_mgard.so"
} > "$BIN/libcompbench_mgard.build-info"
echo "Installed $BIN/libcompbench_mgard.so (temporary sources and build files cleaned on exit)"
