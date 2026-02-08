from __future__ import annotations

import json
import os
import sys

import numpy as np

# Make the repo checkout importable without requiring an editable install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import loopy as lp
from loopy.target.wgsl import get_wgsl_webgpu_info, wgsl_webgpu_info_to_dict


def _artifact_from_kernel(knl: lp.TranslationUnit) -> dict:
    info = get_wgsl_webgpu_info(knl, no_cache=True)
    return wgsl_webgpu_info_to_dict(info)


def make_vector_scale() -> dict:
    # out[i] = 2*a[i]  (bandwidth-heavy, very low compute intensity)
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]",
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    art = _artifact_from_kernel(knl)
    art["meta"] = {
        "id": "vec_scale",
        "title": "Vector Scale",
        "expr": "out[i] = 2*a[i]",
        "dtype": "f32",
        # For this kernel: 1 load (a) + 1 store (out) per element.
        "bytes_per_element": 8,
        "flops_per_element": 1,
        "n_label": "n (elements)",
        "suggested_n": 1 << 20,
        "suggested_iterations": 200,
        "suggested_warmup": 5,
        "suggested_trials": 5,
        "suggested_cpu_trials": 5,
    }
    return art


def make_fma_storm() -> dict:
    # A compute-heavy kernel designed to make the GPU look good (on a real Vulkan/dGPU adapter).
    #
    # Each element does a fixed inner loop with lots of arithmetic but minimal memory traffic:
    #   x = a[i]
    #   out[i] = sum_{k=0..S-1} (
    #       (x + 1*k)*(x - 1*k)
    #     + (x + 2*k)*(x - 2*k)
    #     + (x + 3*k)*(x - 3*k)
    #   )
    #
    # This is deliberately compute-heavy and depends on k so compilers can't hoist all the work.
    # Rough float-FLOPs per k: ~12 (3 mul + ~9 add), plus some integer ops for (m*k).
    inner_steps = 256

    knl = lp.make_kernel(
        "{ [i,k]: 0<=i<n and 0<=k<%d }" % inner_steps,
        [
            "<> x = a[i]",
            "out[i] = sum(k, (x + k)*(x - k) + (x + 2*k)*(x - 2*k) + (x + 3*k)*(x - 3*k))",
        ],
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    art = _artifact_from_kernel(knl)
    art["meta"] = {
        "id": "fma_storm",
        "title": "FMA Storm",
        "expr": f"out[i] = sum(k=0..{inner_steps-1}, (x±k)(x±2k)(x±3k)), x=a[i]",
        "dtype": "f32",
        "bytes_per_element": 8,  # load a[i] once, store out[i] once
        "inner_steps": inner_steps,
        "flops_per_element": 12 * inner_steps,
        "n_label": "n (elements)",
        # Keep default sizes sane: CPU baseline is expensive here.
        "suggested_n": 1 << 18,
        "suggested_iterations": 2,
        "suggested_warmup": 1,
        "suggested_trials": 3,
        "suggested_cpu_trials": 1,
    }
    return art


def make_fma_storm_xl() -> dict:
    # Same idea as make_fma_storm, but with much more work per element to reduce
    # dispatch overhead and make GPU advantages more visible.
    inner_steps = 2048

    knl = lp.make_kernel(
        "{ [i,k]: 0<=i<n and 0<=k<%d }" % inner_steps,
        [
            "<> x = a[i]",
            "out[i] = sum(k, (x + k)*(x - k) + (x + 2*k)*(x - 2*k) + (x + 3*k)*(x - 3*k))",
        ],
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    art = _artifact_from_kernel(knl)
    art["meta"] = {
        "id": "fma_storm_xl",
        "title": "FMA Storm (XL)",
        "expr": f"out[i] = sum(k=0..{inner_steps-1}, (x±k)(x±2k)(x±3k)), x=a[i]",
        "dtype": "f32",
        "bytes_per_element": 8,  # load a[i] once, store out[i] once
        "inner_steps": inner_steps,
        "flops_per_element": 12 * inner_steps,
        "n_label": "n (elements)",
        # Keep defaults conservative: CPU baseline gets expensive fast here.
        "suggested_n": 1 << 15,
        "suggested_iterations": 1,
        "suggested_warmup": 1,
        "suggested_trials": 3,
        "suggested_cpu_trials": 1,
    }
    return art


def make_trig_storm() -> dict:
    # A transcendental-heavy kernel (sin/cos/exp/log) intended to be very slow on
    # scalar JS but often quite good on GPUs. We intentionally do *not* assign a
    # FLOP model here (flops_per_element=0) because "flops" are not meaningful
    # for transcendentals across implementations.
    inner_steps = 64

    knl = lp.make_kernel(
        "{ [i,k]: 0<=i<n and 0<=k<%d }" % inner_steps,
        [
            "<> x = a[i]",
            # Avoid float64 constants in the kernel source (WGSLTarget only supports f32).
            # Keep the expression using integer literals only; mixed int/float arithmetic
            # is handled by the WGSL expression mapper via explicit casts.
            #
            # Keep ranges tame to avoid NaNs/infs across platforms.
            "out[i] = sum(k, "
            "  sin(x + k) * cos(x - 2*k)"
            "  + log(abs(x) + 1 + k)"
            "  + exp(-abs(x) - k)"
            ")",
        ],
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    art = _artifact_from_kernel(knl)
    art["meta"] = {
        "id": "trig_storm",
        "title": "Trig Storm",
        "expr": f"out[i] = sum(k=0..{inner_steps-1}, sin/cos + log + exp), x=a[i]",
        "dtype": "f32",
        "bytes_per_element": 8,
        "inner_steps": inner_steps,
        "flops_per_element": 0,
        "n_label": "n (elements)",
        # Keep defaults safe: JS transcendentals get slow quickly.
        "suggested_n": 1 << 15,
        "suggested_iterations": 1,
        "suggested_warmup": 0,
        "suggested_trials": 3,
        "suggested_cpu_trials": 1,
    }
    return art


def make_matmul() -> dict:
    # Naive square matrix multiply (no tiling):
    #   C[i,j] = sum_k A[i,k] * B[k,j]
    #
    # Arrays are 1D (flattened row-major): idx(i,j) = i*n + j.
    # This is compute-heavy enough that GPUs often win big vs JS, especially on dGPU.
    knl = lp.make_kernel(
        "{ [i,j,k]: 0<=i<n and 0<=j<n and 0<=k<n }",
        "out[i*n + j] = sum(k, a[i*n + k] * b[k*n + j])",
        [lp.GlobalArg("out,a,b", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )
    knl = lp.split_iname(knl, "i", 16, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 16, outer_tag="g.1", inner_tag="l.1")

    art = _artifact_from_kernel(knl)
    art["meta"] = {
        "id": "matmul",
        "title": "Matrix Multiply (naive)",
        "expr": "C = A @ B (square, row-major, no tiling)",
        "dtype": "f32",
        "n_label": "n (matrix side)",
        "suggested_n": 384,
        "suggested_iterations": 3,
        "suggested_warmup": 1,
        "suggested_trials": 3,
        "suggested_cpu_trials": 1,
    }
    return art


def make_matmul_tiled() -> dict:
    # Tiled square matrix multiply using workgroup (shared) memory via add_prefetch.
    #
    # This is a more realistic GPU-friendly matmul than the naive global-memory
    # variant, and should show much better speedups on Vulkan/dGPU adapters.
    tile = 16

    knl = lp.make_kernel(
        "{ [i,j,k]: 0<=i<n and 0<=j<n and 0<=k<n }",
        "out[i, j] = sum(k, a[i, k] * b[k, j])",
        [
            lp.GlobalArg("out", np.float32, shape=("n", "n"), order="C"),
            lp.GlobalArg("a", np.float32, shape=("n", "n"), order="C"),
            lp.GlobalArg("b", np.float32, shape=("n", "n"), order="C"),
            lp.ValueArg("n", np.int32),
        ],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )

    # 2D workgroup: 16x16 threads, each computes one C[i,j] output element.
    knl = lp.split_iname(knl, "i", tile, outer_tag="g.0", inner_tag="l.1", slabs=(0, 1))
    knl = lp.split_iname(knl, "j", tile, outer_tag="g.1", inner_tag="l.0", slabs=(0, 1))
    knl = lp.split_iname(knl, "k", tile, slabs=(0, 1))

    # Prefetch the A and B tiles for each k_outer into workgroup memory.
    knl = lp.add_prefetch(
        knl,
        "a",
        ["k_inner", "i_inner"],
        fetch_outer_inames="i_outer, j_outer, k_outer",
        temporary_address_space=lp.AddressSpace.LOCAL,
        default_tag="l.auto",
    )
    knl = lp.add_prefetch(
        knl,
        "b",
        ["j_inner", "k_inner"],
        fetch_outer_inames="i_outer, j_outer, k_outer",
        temporary_address_space=lp.AddressSpace.LOCAL,
        default_tag="l.auto",
    )

    art = _artifact_from_kernel(knl)
    art["meta"] = {
        "id": "matmul_tiled",
        "title": "Matrix Multiply (tiled, workgroup memory)",
        "expr": f"C = A @ B (square, row-major, {tile}x{tile} tiling, workgroup prefetch)",
        "dtype": "f32",
        "n_label": "n (matrix side)",
        "suggested_n": 384,
        "suggested_iterations": 2,
        "suggested_warmup": 1,
        "suggested_trials": 3,
        "suggested_cpu_trials": 1,
    }
    return art


def main() -> None:
    demos = [
        make_vector_scale(),
        make_fma_storm(),
        make_fma_storm_xl(),
        make_trig_storm(),
        make_matmul(),
        make_matmul_tiled(),
    ]

    payload = {
        "version": 1,
        "demos": demos,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
