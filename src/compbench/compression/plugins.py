"""In-memory native entry points shipped in the pinned hdf5plugin wheel.

SZ and SPERR expose compressor C APIs. SZ3 exposes its C filter callback, obtained
through H5PLget_plugin_info; no HDF5 files, datasets or property lists are used.
API signatures and SZ3 client-data layout follow hdf5plugin 6.0.0 sources.
"""

import ctypes as ct
import struct
from functools import lru_cache
from importlib.metadata import version
from time import perf_counter

import numpy as np

VOID = ct.c_void_p
SIZE = ct.c_size_t
UINT = ct.c_uint
INT = ct.c_int
DOUBLE = ct.c_double
PVOID = ct.POINTER(VOID)
PSIZE = ct.POINTER(SIZE)
FILTER = ct.CFUNCTYPE(SIZE, UINT, SIZE, ct.POINTER(UINT), SIZE, PSIZE, PVOID)


class FilterInfo(ct.Structure):
    _fields_ = [
        ("version", INT),
        ("id", INT),
        ("encoder", UINT),
        ("decoder", UINT),
        ("name", ct.c_char_p),
        ("can_apply", VOID),
        ("set_local", VOID),
        ("filter", FILTER),
    ]


def bind(lib, name, result, *arguments):
    function = getattr(lib, name)
    function.restype = result
    function.argtypes = arguments
    return function


@lru_cache(maxsize=3)
def library(method):
    import hdf5plugin

    if version("hdf5plugin") != "6.0.0":
        raise RuntimeError("Native plugin adapters require hdf5plugin==6.0.0")
    lib = ct.CDLL(hdf5plugin.get_config().registered_filters[method])
    bind(lib, "free", None, VOID)
    if method == "sz":
        bind(lib, "SZ_Init", INT, ct.c_char_p)
        bind(
            lib,
            "SZ_compress_args",
            VOID,
            INT,
            VOID,
            PSIZE,
            INT,
            DOUBLE,
            DOUBLE,
            DOUBLE,
            *([SIZE] * 5),
        )
        bind(lib, "SZ_decompress", VOID, INT, VOID, SIZE, *([SIZE] * 5))
        if lib.SZ_Init(None) != 0:
            raise RuntimeError("SZ initialization failed")
    elif method == "sperr":
        bind(lib, "sperr_comp_2d", INT, VOID, INT, SIZE, SIZE, INT, DOUBLE, INT, PVOID, PSIZE)
        bind(lib, "sperr_decomp_2d", INT, VOID, SIZE, INT, SIZE, SIZE, PVOID)
        bind(lib, "sperr_comp_3d", INT, VOID, INT, *([SIZE] * 6), INT, DOUBLE, SIZE, PVOID, PSIZE)
        bind(lib, "sperr_decomp_3d", INT, VOID, SIZE, INT, SIZE, PSIZE, PSIZE, PSIZE, PVOID)
    else:
        bind(lib, "H5PLget_plugin_info", ct.POINTER(FilterInfo))
        info = lib.H5PLget_plugin_info().contents
        if info.version != 1 or info.id != 32024 or not info.encoder or not info.decoder:
            raise RuntimeError("Unexpected SZ3 filter ABI")
        bind(lib, "malloc", VOID, SIZE)
        # SZ3 returns new[] buffers; its callback consumes malloc-allocated input.
        # Both Linux and macOS builds use the Itanium C++ ABI for operator delete[].
        bind(lib, "_ZdaPv", None, VOID)
        lib.filter = info.filter
    return lib


def array(pointer, shape):
    return (
        np.ctypeslib.as_array(
            ct.cast(pointer, ct.POINTER(ct.c_float)), shape=(int(np.prod(shape)),)
        )
        .copy()
        .reshape(shape)
    )


def sz(x, bound, decode):
    lib = library("sz")
    dims = (0,) * (5 - x.ndim) + x.shape
    size = SIZE()
    pointer = decoded = None
    try:
        start = perf_counter()
        pointer = lib.SZ_compress_args(0, x.ctypes.data, ct.byref(size), 0, bound, 0, 0, *dims)
        comp_time = perf_counter() - start
        if not pointer or not size.value:
            raise RuntimeError("SZ encoding failed")
        result = dict(payload=ct.string_at(pointer, size.value), comp_time=comp_time)
        if decode:
            start = perf_counter()
            decoded = lib.SZ_decompress(0, pointer, size, *dims)
            decomp_time = perf_counter() - start
            if not decoded:
                raise RuntimeError("SZ decoding failed")
            result.update(decoded=array(decoded, x.shape), decomp_time=decomp_time)
        return result
    finally:
        if pointer:
            lib.free(pointer)
        if decoded:
            lib.free(decoded)


def sperr(x, bound, decode):
    lib = library("sperr")
    # This is the original swap=True convention: fastest-varying dimension first.
    dims = x.shape[::-1]
    pointer, decoded, size = VOID(), VOID(), SIZE()
    try:
        start = perf_counter()
        if x.ndim == 2:
            status = lib.sperr_comp_2d(
                x.ctypes.data, 1, *dims, 3, bound, 1, ct.byref(pointer), ct.byref(size)
            )
        else:
            status = lib.sperr_comp_3d(
                x.ctypes.data, 1, *dims, *dims, 3, bound, 1, ct.byref(pointer), ct.byref(size)
            )
        comp_time = perf_counter() - start
        if status or not pointer.value or not size.value:
            raise RuntimeError(f"SPERR encoding failed: {status}")
        result = dict(payload=ct.string_at(pointer, size.value), comp_time=comp_time)
        if decode:
            dx, dy, dz = SIZE(), SIZE(), SIZE()
            start = perf_counter()
            if x.ndim == 2:
                # The native 2D decoder expects the stream without its 10-byte header.
                status = lib.sperr_decomp_2d(
                    pointer.value + 10, size.value - 10, 1, *dims, ct.byref(decoded)
                )
            else:
                status = lib.sperr_decomp_3d(
                    pointer, size, 1, 1, ct.byref(dx), ct.byref(dy), ct.byref(dz), ct.byref(decoded)
                )
            decomp_time = perf_counter() - start
            if status or not decoded.value:
                raise RuntimeError(f"SPERR decoding failed: {status}")
            if x.ndim == 3 and (dx.value, dy.value, dz.value) != dims:
                raise RuntimeError("SPERR decoded shape differs from the input")
            result.update(decoded=array(decoded, x.shape), decomp_time=decomp_time)
        return result
    finally:
        if pointer.value:
            lib.free(pointer)
        if decoded.value:
            lib.free(decoded)


def sz3(x, bound, decode):
    lib = library("sz3")
    # Float32, fastest dimension first, ABS mode, followed by four double bounds.
    words = [
        x.ndim,
        0,
        *x.shape[::-1],
        0,
        *struct.unpack(">8I", struct.pack(">4d", bound, 0, 0, 0)),
    ]
    parameters = (UINT * len(words))(*words)

    def call(address, nbytes, reverse=False):
        original = lib.malloc(nbytes)
        if not original:
            raise MemoryError("SZ3 input allocation failed")
        pointer, capacity = VOID(original), SIZE(nbytes)
        ct.memmove(pointer, address, nbytes)
        try:
            start = perf_counter()
            size = lib.filter(
                256 if reverse else 0,
                len(words),
                parameters,
                nbytes,
                ct.byref(capacity),
                ct.byref(pointer),
            )
            elapsed = perf_counter() - start
            if not size or not pointer.value:
                raise RuntimeError("SZ3 filter failed")
            if reverse:
                if size != x.nbytes:
                    raise RuntimeError("SZ3 decoded size differs from the input")
                return array(pointer, x.shape), elapsed
            return ct.string_at(pointer, size), elapsed
        finally:
            if pointer.value == original:
                lib.free(pointer)
            elif pointer.value:
                lib._ZdaPv(pointer)

    payload, comp_time = call(x.ctypes.data, x.nbytes)
    result = dict(payload=payload, comp_time=comp_time)
    if decode:
        compressed = ct.create_string_buffer(payload)
        decoded, decomp_time = call(compressed, len(payload), reverse=True)
        result.update(decoded=decoded, decomp_time=decomp_time)
    return result
