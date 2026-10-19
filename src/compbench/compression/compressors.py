"""Six compression methods. Inputs are finite float32 arrays; sizes count compressor payloads."""

import ctypes as ct
import os
import shutil
import struct
from contextlib import contextmanager
from functools import lru_cache
from time import perf_counter

import numpy as np

NAMES = ("ebcc", "sz", "sz3", "hpez", "sperr", "mgard")


@contextmanager
def controls(kappa=None, fixed=False, sample=False, random_seed=0):
    values = dict(
        EBCC_KAPPA=None if kappa is None else str(kappa),
        EBCC_INTERNAL_FIXED_QUANTIZATION=str(int(fixed)),
        EBCC_INTERNAL_SAMPLE_RESIDUAL=str(int(sample)),
        EBCC_DISABLE_MCT="0",
    )
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    ct.CDLL(None).srand(ct.c_uint(random_seed))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EBCC:
    def __init__(self):
        from ebcc import EBCC_FILTER_PATH

        self.lib = ct.CDLL(EBCC_FILTER_PATH)
        self.lib.ebcc_encode_field.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_size_t),
            ct.c_float,
            ct.c_float,
            ct.c_double,
            ct.POINTER(ct.c_void_p),
        ]
        self.lib.ebcc_encode_field.restype = ct.c_size_t
        self.lib.ebcc_decode.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_void_p)]
        self.lib.ebcc_decode.restype = ct.c_size_t
        self.lib.free_buffer.argtypes = [ct.c_void_p]
        self.lib.free_buffer.restype = None
        self.lib.ebcc_base_decode_components.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_float,
            ct.c_float,
            ct.POINTER(ct.c_void_p),
            ct.POINTER(ct.c_size_t),
            ct.POINTER(ct.c_size_t),
            ct.c_int,
        ]
        self.lib.ebcc_base_decode_components.restype = ct.c_int
        if hasattr(self.lib, "log_set_level"):
            self.lib.log_set_level(4)

    def encode(self, x, bound):
        dims = (ct.c_size_t * 3)(*(x.shape if x.ndim == 3 else (1, *x.shape)))
        pointer = ct.c_void_p()
        low, high = float(x.min()), float(x.max())
        start = perf_counter()
        size = self.lib.ebcc_encode_field(x.ctypes.data, dims, low, high, bound, ct.byref(pointer))
        elapsed = perf_counter() - start
        if not size or not pointer.value:
            raise RuntimeError("EBCC encoding failed")
        try:
            return ct.string_at(pointer, size), elapsed
        finally:
            self.lib.free_buffer(pointer)

    def decode(self, payload, shape):
        pointer = ct.c_void_p()
        buffer = ct.create_string_buffer(payload)
        start = perf_counter()
        size = self.lib.ebcc_decode(buffer, len(payload), ct.byref(pointer))
        elapsed = perf_counter() - start
        try:
            if size != np.prod(shape) or not pointer.value:
                raise RuntimeError("EBCC returned an incorrect decoded size")
            return (
                np.ctypeslib.as_array(ct.cast(pointer, ct.POINTER(ct.c_float)), shape=(size,))
                .copy()
                .reshape(shape)
            ), elapsed
        finally:
            self.lib.free_buffer(pointer)

    def base(self, payload, shape):
        magic, version, flags, components, low, high, _, step, residual, base_size = struct.unpack(
            "<4sBBHffQdQQ", payload[:48]
        )
        if magic != b"EBCC" or version != 2 or components != 1:
            raise ValueError("Method study needs EBCC v2 single-plane streams")
        raw = payload[48 + residual :]
        if len(raw) != base_size:
            raise ValueError("Invalid EBCC stream sizes")
        pointer, height, width = ct.c_void_p(), ct.c_size_t(), ct.c_size_t()
        ok = self.lib.ebcc_base_decode_components(
            ct.create_string_buffer(raw),
            len(raw),
            1,
            low,
            high,
            ct.byref(pointer),
            ct.byref(height),
            ct.byref(width),
            int(bool(flags & 2)),
        )
        try:
            if not ok or (height.value, width.value) != shape:
                raise RuntimeError("EBCC base reconstruction failed")
            array = (
                np.ctypeslib.as_array(
                    ct.cast(pointer, ct.POINTER(ct.c_float)), shape=(int(np.prod(shape)),)
                )
                .copy()
                .reshape(shape)
            )
        finally:
            self.lib.free_buffer(pointer)
        return (
            array,
            raw,
            dict(
                residual_bytes=residual, base_bytes=base_size, header_bytes=48, residual_step=step
            ),
        )


@lru_cache(maxsize=1)
def ebcc():
    return EBCC()


def executable(name):
    value = os.environ.get(name.upper().replace("-", "_") + "_PATH", name)
    path = shutil.which(value)
    if path is None:
        raise FileNotFoundError(
            f"{name} is unavailable; build the container or set {name.upper().replace('-', '_')}_PATH"
        )
    return path


@lru_cache(maxsize=1)
def hpez_library():
    from compbench.compression.plugins import bind

    lib = ct.CDLL(os.environ.get("HPEZ_LIBRARY_PATH", "libcompbench_hpez.so"))
    bind(lib, "hp_error", ct.c_char_p)
    bind(lib, "hp_free_stream", None, ct.c_void_p)
    bind(lib, "hp_free_field", None, ct.c_void_p)
    bind(
        lib,
        "hp_compress",
        ct.c_int,
        ct.c_void_p,
        ct.c_int,
        ct.POINTER(ct.c_size_t),
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_void_p),
        ct.POINTER(ct.c_size_t),
    )
    bind(
        lib,
        "hp_decompress",
        ct.c_int,
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_void_p),
        ct.POINTER(ct.c_size_t),
    )
    return lib


def hpez(x, bound, decode, cfg):
    from compbench.compression.plugins import array

    lib = hpez_library()
    options = cfg["hpez"]
    target = ("PSNR", "CR", "SSIM", "AC").index(options["tuning_target"])
    level = int(options["level"])
    dims = (ct.c_size_t * x.ndim)(*x.shape)
    pointer, decoded, size, count = ct.c_void_p(), ct.c_void_p(), ct.c_size_t(), ct.c_size_t()
    try:
        start = perf_counter()
        status = lib.hp_compress(
            x.ctypes.data, x.ndim, dims, bound, level, target, ct.byref(pointer), ct.byref(size)
        )
        elapsed = perf_counter() - start
        if status:
            raise RuntimeError(lib.hp_error().decode())
        result = dict(payload=ct.string_at(pointer, size.value), comp_time=elapsed)
        if decode:
            start = perf_counter()
            status = lib.hp_decompress(pointer, size, ct.byref(decoded), ct.byref(count))
            elapsed = perf_counter() - start
            if status:
                raise RuntimeError(lib.hp_error().decode())
            if count.value != x.size:
                raise RuntimeError("HPEZ decoded size differs from the input")
            result.update(decoded=array(decoded, x.shape), decomp_time=elapsed)
        return result
    finally:
        lib.hp_free_stream(pointer)
        lib.hp_free_field(decoded)


@lru_cache(maxsize=1)
def mgard_library():
    path = os.environ.get("MGARD_LIBRARY_PATH", "libcompbench_mgard.so")
    lib = ct.CDLL(path)
    lib.cb_compress.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.POINTER(ct.c_uint64),
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_void_p),
        ct.POINTER(ct.c_size_t),
    ]
    lib.cb_compress.restype = ct.c_int
    lib.cb_decompress.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p]
    lib.cb_decompress.restype = ct.c_int
    lib.cb_free.argtypes = [ct.c_void_p]
    lib.cb_error.restype = ct.c_char_p
    return lib


def mgard(x, bound, decode):
    lib = mgard_library()
    pointer, size = ct.c_void_p(), ct.c_size_t()
    dims = (ct.c_uint64 * x.ndim)(*x.shape)
    try:
        start = perf_counter()
        status = lib.cb_compress(
            x.ctypes.data, x.ndim, dims, bound, 0, 4, 2, ct.byref(pointer), ct.byref(size)
        )
        elapsed = perf_counter() - start
        if status or not pointer.value:
            raise RuntimeError(lib.cb_error().decode())
        result = dict(payload=ct.string_at(pointer, size.value), comp_time=elapsed)
        if decode:
            y = np.empty_like(x)
            start = perf_counter()
            status = lib.cb_decompress(pointer, size, y.ctypes.data)
            result.update(decoded=y, decomp_time=perf_counter() - start)
            if status:
                raise RuntimeError(lib.cb_error().decode())
        return result
    finally:
        if pointer.value:
            lib.cb_free(pointer)


def roundtrip(x, method, bound, cfg, *, decode=True, kappa=None, fixed=False, sample=False):
    x = np.ascontiguousarray(x, dtype=np.float32)
    bound = float(bound)
    if x.ndim not in (2, 3) or not x.size or not np.isfinite(x).all():
        raise ValueError("compressors require nonempty finite 2D/3D float32 input")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("Absolute error bound must be finite and positive")
    if method == "hpez":
        result = hpez(x, bound, decode, cfg)
    elif method == "mgard":
        result = mgard(x, bound, decode)
    elif method == "ebcc":
        compressor = ebcc()
        with controls(kappa, fixed, sample, cfg["random_seed"]):
            payload, comp_time = compressor.encode(x, bound)
        if x.ndim == 2 and payload[5] != 2 * fixed + 4 * sample:
            raise RuntimeError("EBCC library lacks the requested runtime ablation controls")
        result = dict(payload=payload, comp_time=comp_time)
        if decode:
            decoded, decomp_time = compressor.decode(payload, x.shape)
            result.update(decoded=decoded, decomp_time=decomp_time)
    elif method in ("sz", "sz3", "sperr"):
        from compbench.compression import plugins

        result = getattr(plugins, method)(x, bound, decode)
    else:
        raise ValueError(f"Unknown compressor: {method}")
    result["size"] = len(result["payload"])
    return result
