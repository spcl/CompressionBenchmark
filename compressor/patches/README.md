# HPEZ bug fixes for reproduction

Apply `hpez-bugfixes.patch` to public [JLiu-1/HPEZ-QoZ2.0](https://github.com/JLiu-1/HPEZ-QoZ2.0) commit `291699d06d754f58e074d2b2a3a355f45a239e34`. This reproduces the source fixes used by the benchmark without requiring the author's fork.

The patch SHA-256 is `c1b9c70da9b736c018952461439374d28118c778f1ab78c64aa73e8168bd65cf`.

## Scope and provenance

The patch extracts bug fixes from fork commits `0a13e345bd3f1b12878b014192c8dd7a6430c769`, `f2d56ec5b81ae2966c07316907c22b1362ba09de`, and `e6f8176d1eb0489d3e6b26068cb3ec2133da67c6` in [huanglangwen/HPEZ-QoZ2.0](https://github.com/huanglangwen/HPEZ-QoZ2.0). It changes six headers:

| Header under `include/QoZ/` | Fix |
| --- | --- |
| `api/impl/SZInterp.hpp` | Qualify `QoZ::uint` so the casts do not depend on a platform typedef. |
| `utils/Config.hpp` | Include the integer and array declarations used by the header. |
| `compressor/SZGeneralCompressor.hpp` | Account for configuration/frontend/encoder overhead in intermediate buffers and retain the fork's overflow diagnostic. |
| `compressor/SZInterpolationCompressor.hpp` | Account for serialized metadata and Huffman state; initialize the encoder before estimating its size. |
| `lossless/Lossless_zstd.hpp` | Include Zstd's compression bound in allocation and subtract the already-written header from destination capacity. |
| `encoder/HuffmanEncoder.hpp` | Initialize the offset and give copies/moves explicit heap ownership behavior to avoid shared ownership and double frees. |

Version changes, ignore rules, the fork's error report, and whitespace-only edits are excluded. The C ABI wrapper, compiler/linker options, static Zstd integration, and timing/logging configuration remain separate benchmark build choices. The patch does not change quality settings or add performance optimizations.

## Build the reference CLI

Run from the CompressionBenchmark repository root with Git, CMake, a C/C++17 compiler, and the HPEZ build dependencies available. The recipe selects bundled Zstd and disables OpenMP, matching the benchmark's single-thread execution policy.

```bash
benchmark_root="$PWD"
git clone https://github.com/JLiu-1/HPEZ-QoZ2.0.git /tmp/hpez-source
git -C /tmp/hpez-source checkout --detach 291699d06d754f58e074d2b2a3a355f45a239e34
git -C /tmp/hpez-source apply --check "$benchmark_root/compressor/patches/hpez-bugfixes.patch"
git -C /tmp/hpez-source apply "$benchmark_root/compressor/patches/hpez-bugfixes.patch"
cmake -S /tmp/hpez-source -B /tmp/hpez-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DCMAKE_C_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DOMP=OFF -DQoZ_DEBUG_TIMINGS=OFF -DQoZ_USE_BUNDLED_ZSTD=ON
cmake --build /tmp/hpez-build --target hpez --parallel 2
```

The executable is `/tmp/hpez-build/test/hpez`. Use `-q 4 -T SSIM` and the configured per-experiment error mode, bound, and dimensions. Use fresh source/build directories when repeating this recipe. Record the compiler, CMake options, patch hash, bundled Zstd identity, and binary/shared-library hashes with results; this recipe is not itself a portable environment lock.

## Repository build helper

```bash
bash containers/build_hpez.sh
```

The helper checks out the pinned public revision, applies the patch, and links bundled Zstd statically into `compressor/bin/hpez` and `compressor/bin/libcompbench_hpez.so`. The benchmark calls the shared library through `HPEZ_LIBRARY_PATH`; the CLI is retained for reference comparisons. The C ABI wrapper in `compressor/native/hpez.cpp`, build flags and static linking are kept outside the bug-fix patch. `HPEZ_SOURCE_REPO` optionally selects a local Git mirror of the same public revision; `BUILD_JOBS` controls compilation concurrency.

## Validation

Validation uses a clean checkout of the public base from the existing local Git object cache. Patch application succeeds; exactly the six headers above change, and their contents match the intended fork fixes modulo whitespace. Both the CLI and native wrapper build with GCC 16.1.1. Each passes 18 float32 round-trips: smooth and random fields at shapes `(64,96)`, `(3,64,96)`, and `(37,33,48)`, with range-relative targets `0.0001`, `0.01`, and `0.1` converted to absolute bounds, level 4 and SSIM tuning. Reconstruction bounds, shape/dtype, and finite output are checked; the native checks also verify input preservation.

An earlier wrapper and CLI produced different reconstructed values for the smooth `(37,33,48)` field at targets `0.0001` and `0.01`; both satisfy the requested bounds. The other 16 cases match exactly. Those historical checks did not establish interchangeable scientific output. The current benchmark uses the checked-in C ABI wrapper; its verification is recorded below. Earlier CLI measurements must be rerun under measurement version 3.

A focused Huffman ownership check exercises copy/move construction and assignment, self-assignment, reuse, encoding, and destruction under Clang AddressSanitizer without memory-access errors. Leak detection is disabled because the sandbox's tracing prevents LeakSanitizer from running; leak freedom is not established by this check.

These focused checks establish build and small-input behavior. Full scientific datasets and cross-architecture release validation remain part of the refactor plan.

### Current C ABI verification

The checked-in wrapper passes all 18 smooth/random CLI comparisons in
`tests/test_hpez.py`, with identical reconstructed float32 values, valid bounds,
unchanged input, and direct decoding of CLI streams. The same 18 comparisons
pass in the pinned container with networking disabled. The wrapper preserves
the CLI's default `verbose=true`: the pinned upstream source couples that flag
to a dimension-freezing decision, so disabling it is not merely a logging change.
Payload sizes count the actual returned bytes; byte-for-byte stream identity
across different compiler environments is not established by these checks.
