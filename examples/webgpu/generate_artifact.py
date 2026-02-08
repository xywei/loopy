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


def main() -> None:
    # A tiny demo kernel: out[i] = 2 * a[i]
    #
    # This script emits a single JSON artifact that includes:
    # - WGSL source
    # - bind group layout metadata (uniform + storage buffers)
    # - dispatch size expressions (as a small JSON expression language)
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

    # Optional extra metadata for the demo page/UI. The runtime ignores this.
    artifact["meta"] = {
        "title": "Vector Scale",
        "expr": "out[i] = 2*a[i]",
        "dtype": "f32",
        # For this kernel: 1 load (a) + 1 store (out) per element.
        "bytes_per_element": 8,
        "flops_per_element": 1,
        "suggested_n": 1 << 20,
        "suggested_iterations": 200,
    }

    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
