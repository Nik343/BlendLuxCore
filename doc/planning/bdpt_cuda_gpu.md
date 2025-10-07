# CUDA BDPT Enablement Roadmap

## Why
- Current BlendLuxCore exposes LuxCore BDPT only as `BIDIRCPU`; GPU users fall back to standard path tracing which fails for highly specular transport.
- RTX hardware already accelerates unidirectional path tracing via OptiX; extending BDPT closes the feature gap and aligns with community requests.
- Hardware accelerated traversal (RT Cores) plus wavefront scheduling can keep bidirectional throughput competitive with the existing CPU integrator.

## Constraints And Starting Point
- BlendLuxCore is a Blender add-on; BDPT implementation lives inside LuxCoreRender (C++/CUDA). The add-on currently hard locks the engine to CPU (`BIDIRCPU`).
- LuxCore GPU path tracer uses a wavefront OptiX/OpenCL backend with reusable code for ray generation, shading, and film accumulation.
- MIS, vertex storage, and visibility queries remain on programmable shader cores; OptiX/DXR handle BVH traversal.

## Guiding Principles
- Prefer wavefront queues over mega kernels to reduce divergence (matches existing OptiX path tracer design).
- Share as much BSDF sampling, material evaluation, and film accumulation code between CPU and GPU integrators as possible.
- Keep GPU memory footprint bounded: compress vertex records, reuse staging buffers, stream out completed connections.
- Preserve deterministic MIS weighting with shared utility functions across CPU/GPU (avoid double code paths).

## Implementation Phases

### Phase 0 - Prerequisites
- Update build scripts so CUDA + OptiX dependencies are mandatory when `LUXRAYS_ENABLE_CUDA` is ON.
- Add CI job that builds LuxCore with CUDA BDPT enabled and runs smoke regression (cornell box, caustics, volume).
- Extend `pyluxcore` API with capability query (`Session.HasFeature("BIDIR_CUDA")`).

### Phase 1 - GPU Infrastructure (LuxCore core)
- Introduce `BidirGPUPathState` and queue pool mirroring existing `PathOCLState`, storing throughput, MIS data, and vertex buffers for both eye and light subpaths.
- Reuse OptiX acceleration structures: add `BidirOptixPipeline` with ray-gen programs for eye/light extension and shadow visibility.
- Implement persistent kernel loop (`while(activeQueues) { extendEye(); extendLight(); connect(); }`).

### Phase 2 - Eye/Light Subpath Extension Kernels
- Port CPU BDPT vertex sampling logic into GPU callable functions (BSDF sampling, emission, handling volumes).
- Maintain compact vertex arrays (position, normal, throughput, pdfs) allocated via `DeviceBufferPool`; support dynamic length via prefix sums.
- Integrate MIS bookkeeping: store `dVCM`, `dVC`, `dVM` style terms alongside vertices for later connection weights.

### Phase 3 - Connection + MIS Evaluation
- Batched connection kernel: iterate combinations according to current depth limits, run OptiX shadow rays for visibility, compute contribution + MIS weight.
- Support multiple strategies: classic BDPT, light tracing, and optional vertex merging (future VCM reuse).
- Add inline visibility queries via OptiX trace flags to leverage RT cores without switching pipelines.

### Phase 4 - Host Scheduling + Session Interface
- Extend `RenderEngine` selection to accept `BIDIRCUDA` (mirroring `PATHCUDA`).
- Ensure film splatting matches CPU BDPT results; add device to host accumulation path using existing `AddSampleResult` utilities.
- Provide fallbacks: if GPU queue overflows, spill remaining work to CPU or split passes.

### Phase 5 - BlendLuxCore Integration
- Expose `Bidir Device` choice (CPU / CUDA) in UI, gated on capability query.
- Update export pipeline to map `BIDIRCUDA` to correct LuxCore properties and to pass depth parameters.
- Adjust stats strings, viewport helper logic, and automated tests to accept the new engine id.

### Phase 6 - Validation & Optimization
- Create regression scenes targeting specular-diffuse-specular, glossy caustics, volumetric participation.
- Compare CPU vs CUDA BDPT numerically (mean/variance) across 32 seeds.
- Profile queue occupancy and RT core utilization; experiment with batch sizes, persistent threads, and async memory copies.
- Document performance guidelines and known limitations (e.g., GPU memory pressure, OptiX version requirements).

## Task Breakdown Snapshot
- `core/bidir/`: add CUDA path state structs, OptiX pipelines, MIS helpers.
- `lux/core/engines/cuda/`: new `BidirCUDA` engine referencing shared kernels.
- `pyluxcore`: expose engine string, capability query, and device selection enums.
- `BlendLuxCore`: remove CPU lock, export `BIDIR` + device suffix, update UI messaging, extend stats helpers, add tests verifying selection + property roundtrip.

## Open Questions
- How to share existing light vertex caching with CUDA path tracer? Option: reuse `LightStrategy` buffers via device-side indirection.
- Memory budget per path: preliminary estimate 128 bytes per vertex * (eye+light) * maxDepth; verify feasibility for common GPUs (8 GB baseline).
- Scheduling integration with viewport real-time mode - decide if fallback to RT Path or allow lower depth BDPT preview.

## Next Actions
1. Prototype `BidirCUDA` engine in LuxCore core repo (focus on Phase 1 & 2) with a simple scene.
2. Implement capability detection in `pyluxcore` and wire BlendLuxCore UI/export changes (this branch).
3. Iterate on MIS validation vs CPU reference and publish doc + developer guide.
