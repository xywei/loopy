// Loopy WebGPU runner template.
//
// This file is intentionally dependency-free and meant to run in browsers.
// It consumes a JSON artifact produced from loopy's WGSL target (see
// examples/webgpu/generate_artifact.py).
//
// The expected artifact format is the output of:
//   loopy.target.wgsl.wgsl_webgpu_info_to_dict(get_wgsl_webgpu_info(...))
//
// Notes:
// - This is a template, not a production runtime.
// - Buffer usage flags for GPUBuffer inputs are the caller's responsibility.
// - For TypedArray inputs, this runner allocates GPU buffers with STORAGE,
//   COPY_DST, and COPY_SRC to allow both uploads and readback.

export function roundUp(value, multiple) {
  if (!Number.isInteger(multiple) || multiple <= 0) {
    throw new Error(`multiple must be a positive integer, got ${multiple}`);
  }
  return Math.ceil(value / multiple) * multiple;
}

function isArrayBufferView(x) {
  return ArrayBuffer.isView(x) && x.buffer instanceof ArrayBuffer;
}

export async function initWebGPU(
  {
    powerPreference = "high-performance",
    forceFallbackAdapter = false,
    tryFallback = true,
  } = {},
) {
  if (!("gpu" in navigator)) {
    throw new Error(
      "WebGPU is not available (navigator.gpu is missing). Try Chrome/Chromium.",
    );
  }

  const why = [];
  if (typeof isSecureContext === "boolean" && !isSecureContext) {
    why.push(
      "This page is not in a secure context. WebGPU requires HTTPS or localhost.",
    );
  }

  let adapter = await navigator.gpu.requestAdapter({ powerPreference, forceFallbackAdapter });

  // Some systems/browsers only expose a software (fallback) adapter.
  if (!adapter && !forceFallbackAdapter && tryFallback) {
    adapter = await navigator.gpu.requestAdapter({
      powerPreference,
      forceFallbackAdapter: true,
    });
  }

  if (!adapter) {
    const lines = [
      "WebGPU requestAdapter() returned null.",
      ...why,
      "",
      "Troubleshooting:",
      "- Use Chrome/Chromium (known-good baseline for WebGPU).",
      "- Ensure GPU acceleration is enabled in browser settings and restart the browser.",
      "- Check the browser's GPU diagnostics page (e.g. chrome://gpu) for WebGPU/Dawn status.",
      "- If your browser has WebGPU behind a flag, search for 'WebGPU' in its flags/settings and enable it.",
    ];
    throw new Error(lines.join("\n"));
  }

  const device = await adapter.requestDevice();
  return { adapter, device };
}

// Evaluate a small expression language used to represent dispatch sizes.
//
// Expression JSON forms:
// - number: constant
// - {"var": "n"}: variable lookup
// - {"op":"add","args":[...]}
// - {"op":"mul","args":[...]}
// - {"op":"floordiv","a":...,"b":...}
// - {"op":"div","a":...,"b":...}
// - {"op":"mod","a":...,"b":...}
// - {"op":"min","args":[...]}
// - {"op":"max","args":[...]}
export function evalLoopyExpr(expr, vars) {
  if (typeof expr === "number") {
    return expr;
  }

  if (expr && typeof expr === "object" && typeof expr.var === "string") {
    const v = vars[expr.var];
    if (v === undefined) {
      throw new Error(`Missing value for variable '${expr.var}'`);
    }
    return Number(v);
  }

  if (!expr || typeof expr !== "object" || typeof expr.op !== "string") {
    throw new Error(`Invalid expression node: ${JSON.stringify(expr)}`);
  }

  switch (expr.op) {
    case "add": {
      const args = expr.args || [];
      let acc = 0;
      for (const a of args) acc += evalLoopyExpr(a, vars);
      return acc;
    }

    case "mul": {
      const args = expr.args || [];
      let acc = 1;
      for (const a of args) acc *= evalLoopyExpr(a, vars);
      return acc;
    }

    case "floordiv": {
      const a = evalLoopyExpr(expr.a, vars);
      const b = evalLoopyExpr(expr.b, vars);
      return Math.floor(a / b);
    }

    case "div": {
      const a = evalLoopyExpr(expr.a, vars);
      const b = evalLoopyExpr(expr.b, vars);
      return a / b;
    }

    case "mod": {
      const a = evalLoopyExpr(expr.a, vars);
      const b = evalLoopyExpr(expr.b, vars);
      return a % b;
    }

    case "min": {
      const args = expr.args || [];
      if (!args.length) throw new Error("min() requires at least one argument");
      let acc = evalLoopyExpr(args[0], vars);
      for (let i = 1; i < args.length; ++i) acc = Math.min(acc, evalLoopyExpr(args[i], vars));
      return acc;
    }

    case "max": {
      const args = expr.args || [];
      if (!args.length) throw new Error("max() requires at least one argument");
      let acc = evalLoopyExpr(args[0], vars);
      for (let i = 1; i < args.length; ++i) acc = Math.max(acc, evalLoopyExpr(args[i], vars));
      return acc;
    }

    default:
      throw new Error(`Unsupported expr op '${expr.op}'`);
  }
}

function asNonNegativeInt(x, what) {
  const xi = Math.floor(Number(x));
  if (!Number.isFinite(xi) || xi < 0) {
    throw new Error(`${what} must be a finite non-negative integer, got ${x}`);
  }
  return xi;
}

export function packUniformParams(paramsLayout, params) {
  if (!paramsLayout) return null;

  const buf = new ArrayBuffer(paramsLayout.size);
  const view = new DataView(buf);

  for (const field of paramsLayout.fields) {
    if (!(field.name in params)) {
      throw new Error(`Missing uniform param '${field.name}'`);
    }
    const v = params[field.name];

    switch (field.wgsl_type) {
      case "i32":
        view.setInt32(field.offset, v | 0, true);
        break;

      case "u32":
        view.setUint32(field.offset, (v >>> 0), true);
        break;

      case "f32":
        view.setFloat32(field.offset, Number(v), true);
        break;

      default:
        throw new Error(`Unsupported uniform type '${field.wgsl_type}' for field '${field.name}'`);
    }
  }

  return buf;
}

function elementTypeToTypedArrayCtor(elementType) {
  switch (elementType) {
    case "f32":
      return Float32Array;
    case "i32":
      return Int32Array;
    case "u32":
      return Uint32Array;
    default:
      throw new Error(`Unsupported element_type '${elementType}'`);
  }
}

function createBufferFromTypedArray(device, typedArray, usage) {
  if (!isArrayBufferView(typedArray)) {
    throw new Error("Expected an ArrayBufferView (TypedArray) for buffer upload.");
  }

  // WebGPU buffer sizes must be multiples of 4 for many operations.
  const size = Math.max(4, roundUp(typedArray.byteLength, 4));

  const buffer = device.createBuffer({
    size,
    usage,
  });

  device.queue.writeBuffer(buffer, 0, typedArray);
  return buffer;
}

function makeBindGroupLayoutEntries(info) {
  const entries = [];

  if (info.params) {
    entries.push({
      binding: info.params.binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    });
  }

  for (const b of info.storage_bindings) {
    entries.push({
      binding: b.binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: b.webgpu_buffer_binding_type },
    });
  }

  // WebGPU wants these sorted by binding.
  entries.sort((a, b) => a.binding - b.binding);
  return entries;
}

export async function compileLoopyWebGPU(device, artifact) {
  if (!device) throw new Error("device is required");
  if (!artifact || typeof artifact !== "object") throw new Error("artifact is required");
  if (typeof artifact.wgsl_source !== "string") throw new Error("artifact.wgsl_source must be a string");
  if (!Array.isArray(artifact.dispatches)) throw new Error("artifact.dispatches must be an array");

  const t0 = performance.now();

  const label = (artifact.meta && typeof artifact.meta.id === "string" && artifact.meta.id.trim())
    ? `loopy:${artifact.meta.id.trim()}`
    : "loopy";
  const module = device.createShaderModule({ label, code: artifact.wgsl_source });

  // Best-effort compilation diagnostics (Chrome/Chromium supports this).
  let compilationInfo = null;
  if (module && typeof module.getCompilationInfo === "function") {
    try {
      compilationInfo = await module.getCompilationInfo();
    } catch {
      compilationInfo = null;
    }
  }

  const bgl = device.createBindGroupLayout({
    entries: makeBindGroupLayoutEntries(artifact),
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bgl],
  });

  const pipelines = new Map();
  for (const d of artifact.dispatches) {
    if (pipelines.has(d.entrypoint)) continue;
    const desc = {
      layout: pipelineLayout,
      compute: { module, entryPoint: d.entrypoint },
    };

    // Wrap pipeline creation in an error scope so we can surface validation errors
    // with a useful message (instead of "[Invalid ShaderModule] is invalid.").
    // https://www.w3.org/TR/webgpu/#error-scopes
    let pipeline = null;
    let scopeErr = null;
    let pushed = false;
    try {
      if (typeof device.pushErrorScope === "function" && typeof device.popErrorScope === "function") {
        device.pushErrorScope("validation");
        pushed = true;
      }

      pipeline = device.createComputePipelineAsync
        ? await device.createComputePipelineAsync(desc)
        : device.createComputePipeline(desc);
    } finally {
      if (pushed && typeof device.popErrorScope === "function") {
        try {
          scopeErr = await device.popErrorScope();
        } catch {
          scopeErr = null;
        }
      }
    }

    if (scopeErr) {
      const lines = [
        `WebGPU pipeline creation failed for entryPoint='${d.entrypoint}'.`,
        scopeErr && scopeErr.message ? String(scopeErr.message) : String(scopeErr),
      ];

      if (compilationInfo && Array.isArray(compilationInfo.messages) && compilationInfo.messages.length) {
        lines.push("");
        lines.push("WGSL compilation messages:");
        for (const m of compilationInfo.messages) {
          const loc = (m && m.lineNum && m.linePos)
            ? `:${m.lineNum}:${m.linePos}`
            : "";
          const kind = m && m.type ? String(m.type) : "info";
          const msg = m && m.message ? String(m.message) : String(m);
          lines.push(`- ${kind}${loc}: ${msg}`);
        }
      }

      throw new Error(lines.join("\n"));
    }

    pipelines.set(d.entrypoint, pipeline);
  }

  const t1 = performance.now();

  return {
    device,
    artifact,
    module,
    compilationInfo,
    bgl,
    pipelineLayout,
    pipelines,
    timings: {
      compileMs: t1 - t0,
    },
  };
}

export function createLoopyBindGroup(
  device,
  compiled,
  {
    params = {},
    buffers = {},
    bufferByteSizes = {},
  } = {},
) {
  if (!device) throw new Error("device is required");
  if (!compiled || typeof compiled !== "object") throw new Error("compiled is required");

  const { artifact, bgl } = compiled;

  // Create/upload uniform params if present.
  let uniformBuffer = null;
  if (artifact.params) {
    const packed = packUniformParams(artifact.params, params);
    uniformBuffer = device.createBuffer({
      size: Math.max(16, roundUp(artifact.params.size, 4)),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, packed);
  }

  // Create/upload storage buffers.
  const storageBuffers = {};
  let bytesUploaded = 0;
  for (const b of artifact.storage_bindings) {
    const userBuf = buffers[b.name];
    if (!userBuf) {
      throw new Error(`Missing buffer for '${b.name}'`);
    }

    if (userBuf instanceof GPUBuffer) {
      storageBuffers[b.name] = userBuf;
      continue;
    }

    if (isArrayBufferView(userBuf)) {
      storageBuffers[b.name] = createBufferFromTypedArray(
        device,
        userBuf,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      );
      bytesUploaded += userBuf.byteLength;
      continue;
    }

    throw new Error(`Buffer for '${b.name}' must be a GPUBuffer or a TypedArray`);
  }

  // Create bind group.
  const bindEntries = [];
  if (artifact.params) {
    bindEntries.push({
      binding: artifact.params.binding,
      resource: { buffer: uniformBuffer },
    });
  }
  for (const b of artifact.storage_bindings) {
    bindEntries.push({
      binding: b.binding,
      resource: { buffer: storageBuffers[b.name] },
    });
  }
  bindEntries.sort((a, b) => a.binding - b.binding);

  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: bindEntries,
  });

  return {
    uniformBuffer,
    storageBuffers,
    bindGroup,
    bytesUploaded,
    bufferByteSizes,
  };
}

export function evaluateDispatches(artifact, params) {
  const evaluated = [];
  const vars = { ...params };
  for (const d of artifact.dispatches) {
    const x = asNonNegativeInt(evalLoopyExpr(d.workgroups[0], vars), `${d.entrypoint}.workgroups.x`);
    const y = asNonNegativeInt(evalLoopyExpr(d.workgroups[1], vars), `${d.entrypoint}.workgroups.y`);
    const z = asNonNegativeInt(evalLoopyExpr(d.workgroups[2], vars), `${d.entrypoint}.workgroups.z`);
    evaluated.push({
      entrypoint: d.entrypoint,
      workgroup_size: d.workgroup_size,
      workgroups: [x, y, z],
    });
  }
  return evaluated;
}

export async function dispatchLoopyWebGPU(
  device,
  compiled,
  resources,
  {
    params = {},
    iterations = 1,
  } = {},
) {
  if (!device) throw new Error("device is required");
  if (!compiled || typeof compiled !== "object") throw new Error("compiled is required");
  if (!resources || typeof resources !== "object") throw new Error("resources is required");

  const { artifact, pipelines } = compiled;

  if (artifact.params && resources.uniformBuffer) {
    const packed = packUniformParams(artifact.params, params);
    device.queue.writeBuffer(resources.uniformBuffer, 0, packed);
  }

  const dispatches = evaluateDispatches(artifact, params);

  // Pre-resolve pipelines and skip zero-sized dispatches.
  const dispatchCalls = [];
  for (const d of dispatches) {
    const [x, y, z] = d.workgroups;
    if (x === 0 || y === 0 || z === 0) continue;

    const pipeline = pipelines.get(d.entrypoint);
    if (!pipeline) throw new Error(`internal error: missing pipeline for '${d.entrypoint}'`);

    dispatchCalls.push({
      entrypoint: d.entrypoint,
      pipeline,
      x,
      y,
      z,
    });
  }

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setBindGroup(0, resources.bindGroup);

  let currentPipeline = null;
  for (let it = 0; it < iterations; ++it) {
    for (const call of dispatchCalls) {
      if (call.pipeline !== currentPipeline) {
        pass.setPipeline(call.pipeline);
        currentPipeline = call.pipeline;
      }
      pass.dispatchWorkgroups(call.x, call.y, call.z);
    }
  }

  pass.end();

  const t0 = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const t1 = performance.now();

  return {
    dispatches,
    iterations,
    submitToDoneMs: t1 - t0,
  };
}

export async function readbackLoopyBuffers(
  device,
  compiled,
  resources,
  {
    buffers = {},
    bufferByteSizes = {},
    names = [],
  } = {},
) {
  if (!device) throw new Error("device is required");
  if (!compiled || typeof compiled !== "object") throw new Error("compiled is required");
  if (!resources || typeof resources !== "object") throw new Error("resources is required");

  const { artifact } = compiled;

  // Prepare readback buffers (COPY_DST | MAP_READ).
  const readbackPlans = [];
  const readbackBuffers = new Map();
  for (const name of names) {
    const binding = artifact.storage_bindings.find((b) => b.name === name);
    if (!binding) {
      throw new Error(`readback requested for unknown buffer '${name}'`);
    }

    const userBuf = buffers[name];
    const sizeBytes = isArrayBufferView(userBuf)
      ? userBuf.byteLength
      : bufferByteSizes[name];

    if (!Number.isInteger(sizeBytes) || sizeBytes <= 0) {
      throw new Error(
        `readback for '${name}' requires a TypedArray in buffers['${name}'] or bufferByteSizes['${name}']`,
      );
    }
    if (sizeBytes % 4 !== 0) {
      throw new Error(`readback byte size for '${name}' must be a multiple of 4 (got ${sizeBytes})`);
    }

    const rb = device.createBuffer({
      size: Math.max(4, roundUp(sizeBytes, 4)),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    readbackBuffers.set(name, rb);
    readbackPlans.push({
      name,
      sizeBytes,
      element_type: binding.element_type,
      element_nbytes: binding.element_nbytes,
    });
  }

  const encoder = device.createCommandEncoder();
  for (const plan of readbackPlans) {
    const src = resources.storageBuffers[plan.name];
    const dst = readbackBuffers.get(plan.name);
    encoder.copyBufferToBuffer(src, 0, dst, 0, plan.sizeBytes);
  }

  const t0 = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const t1 = performance.now();

  // Materialize readback.
  const readbackResults = {};
  const tMap0 = performance.now();
  for (const plan of readbackPlans) {
    const rb = readbackBuffers.get(plan.name);
    await rb.mapAsync(GPUMapMode.READ);
    const mapped = rb.getMappedRange();

    const userBuf = buffers[plan.name];
    if (isArrayBufferView(userBuf)) {
      // Copy bytes into the caller's TypedArray in-place.
      const dstBytes = new Uint8Array(userBuf.buffer, userBuf.byteOffset, userBuf.byteLength);
      dstBytes.set(new Uint8Array(mapped, 0, dstBytes.byteLength));
      readbackResults[plan.name] = userBuf;
    } else {
      const Ctor = elementTypeToTypedArrayCtor(plan.element_type);
      const count = plan.sizeBytes / plan.element_nbytes;
      const out = new Ctor(count);
      new Uint8Array(out.buffer).set(new Uint8Array(mapped, 0, plan.sizeBytes));
      readbackResults[plan.name] = out;
    }

    rb.unmap();
  }
  const tMap1 = performance.now();

  return {
    readback: readbackResults,
    timings: {
      copySubmitToDoneMs: t1 - t0,
      mapAndCopyMs: tMap1 - tMap0,
    },
  };
}

export async function runLoopyWebGPU(
  device,
  artifact,
  {
    params = {},
    buffers = {},
    bufferByteSizes = {},
    readback = [],
    iterations = 1,
  } = {},
) {
  const compiled = await compileLoopyWebGPU(device, artifact);
  const resources = createLoopyBindGroup(device, compiled, { params, buffers, bufferByteSizes });

  const dispatchResult = await dispatchLoopyWebGPU(device, compiled, resources, { params, iterations });

  const rb = readback.length
    ? await readbackLoopyBuffers(device, compiled, resources, { buffers, bufferByteSizes, names: readback })
    : { readback: {}, timings: { copySubmitToDoneMs: 0, mapAndCopyMs: 0 } };

  return {
    compiled,
    resources,
    dispatch: dispatchResult,
    readback: rb.readback,
    timings: {
      compileMs: compiled.timings.compileMs,
      dispatchSubmitToDoneMs: dispatchResult.submitToDoneMs,
      readbackCopySubmitToDoneMs: rb.timings.copySubmitToDoneMs,
      readbackMapAndCopyMs: rb.timings.mapAndCopyMs,
    },
  };
}
