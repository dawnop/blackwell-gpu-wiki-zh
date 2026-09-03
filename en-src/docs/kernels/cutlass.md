# CUTLASS

NVIDIA's CUDA Templates library. The reference C++ template library for high-performance GEMM (and now also convolutions and conv-related ops). Most other GPU kernel libraries are either built on CUTLASS or directly inspired by it.

## What it is

A header-only C++ library of templates that compile to optimized CUDA kernels for matrix multiplication. The user instantiates a template specifying:

- Operand types (BF16, FP8, NVFP4, etc.)
- Tile shape (e.g., m128n128k64)
- Pipeline stages
- Target architecture (e.g., `cutlass::arch::Sm100`, `cutlass::arch::Sm120`)
- Layout (row/col-major)

CUTLASS then generates a kernel tuned for that combination. Modern (CUTLASS 3.x) uses CUTE — a high-level layout/algebra library — under the hood for clean indexing.

GitHub: `NVIDIA/cutlass`. Maintained by NVIDIA. Used as the GEMM backend by FlashInfer, vLLM (in some paths), TensorRT-LLM, sglang, DeepSeek-AI's stack, and many others.

## What it depends on

- CUDA toolkit (CUDA ≥ 12.8 for SM100; supported since CUTLASS 3.8)
- C++17
- A C++ compiler that nvcc can drive

CUTLASS itself depends on nothing else at runtime — it's header-only, compiled into whatever uses it.

## SM100 story

CUTLASS 3.8+ has dedicated Blackwell datacenter templates under `cutlass/include/cutlass/gemm/collective/sm100_*` and `cutlass/include/cutlass/gemm/kernel/sm100_*`. These templates:

- Target `sm_100a` (architecture-specific accelerated)
- Use `tcgen05.mma` for the inner MMA loop
- Allocate accumulators in TMEM via `tcgen05.alloc`
- Use **single-CTA mode** (`cta_group::1`) by default; CTA-pair mode (`cta_group::2`) is opt-in via tile-shape choice
- Request up to ~220 KiB of SMEM for operand-staging pipeline buffers
- Lean on cluster-shared TMA for cross-CTA data movement when CTA-pair mode is on

Compiling them with `nvcc -gencode arch=compute_100,code=sm_100a` produces a fatbin that runs only on SM 10.0 devices.

## SM120 story

CUTLASS 3.6+ has parallel templates under `sm120_*`. These templates:

- Target `sm_120` (or `sm_120f` for forward-compat)
- Use `mma.sync` (including the `sm_120a`-only block-scaled variants) instead of `tcgen05.mma`
- Allocate accumulators in **registers** (smaller tiles to fit) or stage through SMEM (larger tiles, more SMEM pressure)
- Use **single-CTA only** (no `cluster_dim > 1`)
- Restrict pipeline stages to fit the 99 KiB SMEM ceiling
- Achieve ~40–70 % of optimal SM120 throughput vs ~95 % for SM100 templates on SM100

The SM120 templates are *separate trees* from the SM100 templates. They're not just a recompile.

## The SMEM cliff

The single most-encountered CUTLASS issue on workstation Blackwell. The story:

1. CUTLASS uses `StageCountAutoCarveout<sizeof(SharedStorage)>` to determine how many pipeline stages to fit in available SMEM.
2. `StageCountAutoCarveout` calculates remaining SMEM as `total_smem - other_uses` where `total_smem` is taken from the architecture's published max.
3. On SM100, max is 228 KiB. On SM120, max is **99 KiB**.
4. If a developer tests their template on SM100 (with 228 KiB headroom) and then runs the same code on SM120, the auto-carveout calculation believes there's 228 KiB available and requests pipeline buffers that overflow the actual 99 KiB.
5. `cudaFuncSetAttribute` returns `cudaErrorInvalidValue` for anything over 99 KiB, and a direct launch fails with out-of-resources. The error only appears at runtime, nothing warns at compile time, and higher-level libraries often swallow it into a vague "kernel unavailable".

The canonical issue: `NVIDIA/cutlass#3096` ("SMEM size detection on Blackwell consumer parts"). The fix in flight: a runtime SMEM budget query that respects the actual device limits.

Workarounds in the meantime:

- Manually set `StageCount` to a small number (2 or 3) instead of auto-carveout
- Use the `sm120_*` templates rather than the `sm100_*` ones
- Choose smaller tile shapes that don't push SMEM hard

## What a CUTLASS SM100 kernel looks like

Coming from the Hopper `sm90_*` templates, both the names and the division of labour change:

- Kernel schedules: `KernelTmaWarpSpecialized1SmSm100` / `2SmSm100`; block-scaled `...{1,2}SmBlockScaledSm100`, further specialised as `Nvf4 / Mxf4 / Mxf8f6f4`. `KernelScheduleAuto` picks 1SM or 2SM from the cluster shape
- Mainloop: `MainloopSm100TmaUmmaWarpSpecialized`
- Epilogue: `Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>`, reading the accumulator out of TMEM with copy atoms such as `SM100_TMEM_LOAD_16dp256b1x / 32dp32b1x`
- Reference warp roles: warp 0 = MMA (one lane issues), warp 1 = scheduler (CLC), warp 2 = TMA load, warp 3 = epilogue load, warp 4 onwards = epilogue (four warps by default, each reading its own quarter of the lanes). Hopper's "two consumer warpgroups each holding accumulator registers" model is gone: one thread issues MMA, one thread issues TMA, and copying the Hopper layout wastes warps and registers
- Versions: 3.8 first supports SM100; 4.0 (June 2025) adds the CuTe DSL; 4.2 adds SM103; 4.5 (May 2026) adds mixed-precision MXF8F6F4; 4.7.1 is the stable release as of September 2026

Entry point in the source: `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`.

## Common failures

**Failure 1: `no kernel image is available`**

You built a CUTLASS-using library against `sm_100a` and ran it on SM 12.0. The fatbin contains only `sm_100a` cubins, no SASS for `sm_120`, and the embedded PTX (if any) targets `sm_100a` which can't JIT to `sm_120`.

Fix: rebuild with `-gencode arch=compute_120,code=sm_120` *and* the SM120-targeted templates, not just the SM100 templates with a different gencode flag.

**Failure 2: SMEM over the limit, launch fails (the SMEM cliff)**

Described above. The symptom is `cudaErrorInvalidValue` or a launch out-of-resources error, often wrapped by higher-level libraries into "kernel unavailable".

Detection: `nvcc --ptxas-options=-v` shows the SMEM footprint of the template instance; check whether it exceeds 99 KiB. Or check `cudaGetLastError` around the launch.

Fix: use SM120 templates with smaller tile shapes, or set explicit `StageCount`.

**Failure 3: CTA-pair MMA on SM120**

A CUTLASS template with `cta_group::2` (CTA-pair MMA) is built for SM120. `ptxas` rejects the `tcgen05.*` instructions at compile time; a precompiled `sm_100a` cubin fails at load with "no kernel image is available". SM120 does support clusters — the failure is the `tcgen05` dependency, and it happens before launch.

Fix: only use CUTLASS templates that have `cta_group::1`. The SM120 template tree enforces this; the SM100 tree does not.

**Failure 4: NVFP4 scale layout mismatch**

CUTLASS expects NVFP4 scales in a specific layout (block-interleaved, FP8 E4M3). If a model artifact was saved in MX-FP4 layout (block-32 with E8M0 scales) and loaded into a CUTLASS NVFP4 template, the scales are misinterpreted.

Fix: requantize the artifact, or use a different kernel library whose layout matches.

## Detection

To check whether a `.so` uses CUTLASS:

```bash
nm -D mylib.so | grep -i cutlass | head
# or
strings mylib.so | grep -E 'cutlass::|CollectiveBuilder|StageCount' | head
```

To check which arch targets are present:

```bash
cuobjdump --list-elf mylib.so
```

Look for `arch = sm_100a` (datacenter only) or `arch = sm_120` (workstation friendly).

## Reading CUTLASS source

The library is large (~200K LOC) but well-organized:

```
include/cutlass/
├── gemm/
│   ├── collective/
│   │   ├── sm70_*       # Volta
│   │   ├── sm80_*       # Ampere
│   │   ├── sm90_*       # Hopper
│   │   ├── sm100_*      # Blackwell datacenter
│   │   └── sm120_*      # Blackwell workstation
│   ├── kernel/          # Top-level kernel composition
│   └── threadblock/     # Older (pre-3.x) tile-level code
├── arch/                # Architecture wrappers (cutlass::arch::Sm120 etc.)
├── conv/                # Convolutions (similar structure)
└── ...
```

To understand what's specific to one architecture, diff `sm100_*` against `sm120_*`. The differences will be in MMA-instruction wrappers, tile shapes, and pipeline depth.

## CUTLASS issues to watch

A few open / recent issues that capture the SM120 story:

- `#3096` — SMEM size detection on consumer Blackwell
- `#3045` — NVFP4 scale layout discrepancies
- `#2950` — sm120 template stagecount auto-carveout
- `#3120` — wgmma fallback path for sm120

These issues are the current edge of CUTLASS development; their resolution will affect what works on SM120 in subsequent releases.

## See also

- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) — `mma.sync`, `wgmma`, `tcgen05`
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — what the `sm100_*` templates use
- [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md) — the SMEM cliff
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) — porting patterns
- *NVIDIA/cutlass* on GitHub
- *CUTLASS Programming Guide* (in the repo's `media/docs/` directory)
