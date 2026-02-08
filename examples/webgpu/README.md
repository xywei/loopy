# loopy + WebGPU (WGSL) demo

This folder contains a minimal, browser-side runner template for code generated
by loopy's `WGSLTarget`.

## What you get

- `loopy-webgpu-runner.js`: dependency-free WebGPU runner that:
  - builds bind group layout from loopy's metadata
  - packs scalar params into a uniform buffer
  - uploads storage buffers
  - dispatches one or more compute entrypoints (for kernels split at global barriers)
  - optionally reads back selected buffers
- `generate_artifact.py`: generates `artifact.json` for a tiny demo kernel.
- `index.html`: loads `artifact.json` and runs the demo.

## Quickstart

1. Generate artifacts:
   ```sh
   cd examples/webgpu
   LOOPY_NO_CACHE=1 python generate_artifacts.py > artifacts.json
   ```

2. Start a local server (browsers typically block `fetch()` from `file://`):
   ```sh
   python -m http.server 8000
   ```

3. Open:
   - `http://localhost:8000/index.html`

4. Optional:
   - Toggle "Read back + verify output" to copy results back to the CPU and
     validate correctness (this adds overhead, so it's off by default).

## Notes

- WebGPU is still gated on some platforms/browsers. Chrome/Chromium are the
  easiest place to start.
- If you pass `GPUBuffer` objects into the runner, make sure they were created
  with the usage flags needed by your flow (at minimum `GPUBufferUsage.STORAGE`,
  plus `COPY_DST` for uploads and `COPY_SRC` for readback).

## Troubleshooting: `requestAdapter()` returned null

If you see `WebGPU requestAdapter() returned null`, the WGSL codegen is not the
problem yet. It means the browser couldn't provide a WebGPU adapter.

Things to check:

- Browser: try Chrome/Chromium first.
- Hardware acceleration: ensure it's enabled in browser settings, then restart.
- Diagnostics: in Chromium browsers, open `chrome://gpu` and look for WebGPU/Dawn status.
- Context: WebGPU generally requires a secure context (HTTPS or `http://localhost`).
- Flags: if WebGPU is behind a flag in your browser build, search for "WebGPU" in its flags/settings and enable it.
