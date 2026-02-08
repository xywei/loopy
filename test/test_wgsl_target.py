from __future__ import annotations


import numpy as np

import loopy as lp
from loopy.target.wgsl import get_wgsl_webgpu_info, wgsl_webgpu_info_to_dict


def test_wgsl_target_generates_compute_shader() -> None:
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]",
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    with lp.CacheMode(False):
        code = lp.generate_code_v2(knl).device_code()

    assert "@compute" in code
    assert "@workgroup_size(128" in code

    # Resource interface
    assert "var<uniform> _lpy_params: LoopyParams" in code
    assert "var<storage" in code
    assert "struct LoopyBuf_out" in code
    assert "struct LoopyBuf_a" in code

    # Value args routed through uniform params
    assert "_lpy_params.n" in code

    # Array args routed through .data
    assert "out.data)[" in code
    assert "a.data)[" in code


def test_wgsl_target_emits_workgroup_temporaries() -> None:
    knl = lp.make_kernel(
        "{ [g, l]: 0<=g<ngroups and 0<=l<128 }",
        [
            "<> tmp[l] = a[128*g + l]",
            "out[128*g + l] = tmp[l]",
        ],
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )

    knl = lp.tag_inames(knl, {"g": "g.0", "l": "l.0"})
    knl = lp.set_temporary_address_space(knl, "tmp", "local")

    with lp.CacheMode(False):
        code = lp.generate_code_v2(knl).device_code()

    # LOCAL temporaries become module-scope workgroup variables.
    assert "var<workgroup> tmp: array<f32, 128>;" in code
    assert "var tmp:" not in code


def test_wgsl_webgpu_info_is_json_serializable() -> None:
    import json

    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]",
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )

    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    info = get_wgsl_webgpu_info(knl, no_cache=True)
    artifact = wgsl_webgpu_info_to_dict(info)
    json.dumps(artifact)


def test_wgsl_target_casts_inames_in_float_expressions() -> None:
    # WGSL does not allow mixed-type arithmetic like f32 + i32. Ensure that
    # integer inames get cast when used in float contexts.
    knl = lp.make_kernel(
        "{[i,k]: 0<=i<n and 0<=k<4}",
        "out[i] = sum(k, a[i] + k)",
        [lp.GlobalArg("out,a", np.float32, shape=lp.auto), "..."],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    with lp.CacheMode(False):
        code = lp.generate_code_v2(knl).device_code()

    assert "f32(k)" in code


def test_wgsl_target_emits_integer_floor_div_helpers() -> None:
    # Loopy uses late-bound helper functions for floor division/modulo to match
    # mathematical semantics (as opposed to truncation-toward-zero i32 division).
    # Ensure WGSLTarget provides these helpers when they are referenced.
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = a[(-1 + n)//16]",
        [
            lp.GlobalArg("out", np.float32, shape=("n",)),
            lp.GlobalArg("a", np.float32, shape=("n",)),
            lp.ValueArg("n", np.int32),
        ],
        target=lp.WGSLTarget(),
        lang_version=(2018, 2),
    )
    knl = lp.split_iname(knl, "i", 64, outer_tag="g.0", inner_tag="l.0")

    with lp.CacheMode(False):
        code = lp.generate_code_v2(knl).device_code()

    assert "loopy_floor_div_pos_b_int32" in code
    assert "fn loopy_floor_div_pos_b_int32" in code
