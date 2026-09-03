# Porting from Hopper to B200

The rest of this wiki is about moving SM100 software down to SM120. This page goes the other way: you have kernels that run well on H100 (`sm_90a`) and need them on B200 (`sm_100a`). Ordered from "does not compile" to "runs slowly". Sources at the end.

## 1. Binaries: nothing carries over

- Not a single line of `wgmma` carries over, and there is no fallback. `wgmma.mma_async / fence / commit_group / wait_group` are all "Requires sm_90a" in the PTX ISA.
- Neither `sm_90a` cubins nor `sm_90a` PTX load; `sm_90` cubins do not load either (9 → 10 is a major-version change); only PTX without an `a` suffix can be JIT-compiled. Check with `CUDA_FORCE_PTX_JIT=1`.
- Building for `sm_100` means no `tcgen05`. Use `sm_100a`, or `sm_100f` if one binary should also cover B300. PyTorch extensions must say `10.0a` explicitly.
- Minimum CUDA 12.8 and driver 570.26; the `f` suffix needs 12.9. CUDA 13.0 removed `cudaDeviceProp.clockRate` and friends, which breaks old harnesses.
- `mma.sync` is the only MMA path that runs without a rewrite, but FP8 through it on B200 is an HMMA on FP16-converted inputs.

Details in [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md).

## 2. Do not use RTX 50 numbers for B200

arXiv 2507.10789 measured an RTX 5080 (SM120); its L2, SMEM and `mma.sync` numbers do not apply to B200. The B200 microbenchmark paper is arXiv 2512.02189. Every throughput number on the data sheets is sparse, so divide by 2 for dense; HGX B200 and the GPU inside GB200 are two different bins. See [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md).

## 3. TMEM is a runtime resource; static occupancy cannot see it

- `tcgen05.alloc` **blocks** rather than fails when TMEM is short; a second CTA on the same SM asking for 512 columns waits forever.
- Call `relinquish_alloc_permit` right after alloc; a later alloc may not be larger than an earlier one; `dealloc` before exit.
- `cudaOccupancy*` knows nothing about TMEM. How many CTAs fit on an SM is decided at runtime by the columns you allocate.

See [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md).

## 4. Rearrange the warp roles

Hopper's "one producer warpgroup plus two consumer warpgroups each holding accumulator registers" model is gone:

- `tcgen05.mma` is issued by **one thread**, and so is TMA. Copying the Hopper layout wastes warps and registers.
- The epilogue needs at least four warps; each can read only its own 32 TMEM lanes, so no single warp sees the whole accumulator.
- Completion comes in two mechanisms: `wait::ld / st` for `tcgen05.ld / st`, `commit` plus mbarrier for `mma / cp / shift`. They do not mix.
- Every `tcgen05` instruction in the kernel must use the same `.cta_group`.
- CUTLASS's reference roles: warp 0 MMA, warp 1 scheduler, warp 2 TMA, warp 3 epilogue load, warp 4 onwards epilogue. See [`kernels/cutlass`](../kernels/cutlass.md).

## 5. Shape and data-type traps

- The block-scaled kinds (`mxf8f6f4 / mxf4 / mxf4nvf4`) have no M=64, only M=128.
- `mxf4 / mxf4nvf4 / i8` are `sm_100a` only; `sm_100f` does not have them.
- Scales must be copied into TMEM with `tcgen05.cp` first, and the GMEM layout CUTLASS defines is fixed and cannot be transposed.
- In `kind::f8f6f4` every 4- and 6-bit element occupies a full byte; only the `mxf4` kinds are truly packed.
- INT4 does not exist in `tcgen05`.
- See [`fundamentals/number-formats`](../fundamentals/number-formats.md).

## 6. FP8 no longer needs promotion

Hopper's trick of returning to CUDA cores every 128 K to accumulate is wasted work on B200: `tcgen05` FP8 accumulation measures 25 mantissa bits. The scale format does change, to packed UE8M0. DeepGEMM's SM100 kernels made exactly this change; see [`kernels/deepgemm`](../kernels/deepgemm.md).

## 7. Roles inside a 2-SM MMA

- B is the shared operand: each CTA stages only half of N; A and D stay private to each CTA.
- The cluster must have an even number of CTAs; only the leader issues the MMA.
- TMA completion is signalled to the partner CTA's mbarrier with `.cta_group::2`; `tcgen05.commit` multicasts to both.
- See [`blackwell/thread-block-clusters`](../blackwell/thread-block-clusters.md).

## 8. Clocks, power and measurement

- NVIDIA does not publish B200's clock. Working back from the rated throughput gives roughly 1.9 GHz for HGX B200 and 2.1 GHz for GB200; people have recorded a 1.965 GHz ceiling and a 700 MHz base in nvidia-smi. None of this is official.
- At the power cap the GPU runs DVFS, and locking the clock does not hold it. CUTLASS's performance measurement guide recommends zero-filled data, monitoring the clock, and long warm-up plus cool-down (for large Blackwell GEMMs: 10000 warm-up iterations, 4000 measured, 1 s cool-down).
- HGX B200 can be configured up to 1000 W, GB200 up to 1200 W; DGX B200 caps at 700 W by default.

## 9. L2 partitions

The 126 MB L2 is split into partitions; cross-partition bandwidth drops from about 21 TB/s to about 17 TB/s and latency rises noticeably. Third parties disagree on the partition count (2 or 4), NVIDIA has no documentation, and there is no SM-to-partition mapping API. See [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md).

## 10. Tooling

- ncu supports Blackwell from 2024.4. The Profiling Guide defines three pipeline groups: `tc` (UTCBAR / UTCCP / UTC\*MMA / UTCSHIFT), `tmem` (LDT / STT), `tma`.
- The decompression engine (Snappy / LZ4 / Deflate / Gzip, up to 600 GB/s) is host-side only, through the driver API `cuMemBatchDecompressAsync`; there is no device-side PTX interface. It is unrelated to the FP4 / FP6 unpacking in `tcgen05.cp`.
- Other new `sm_100` PTX: `redux.sync.f32`; 256-bit `ld/st .v8.b32` and `.level2::eviction_priority` (8.8); `ldmatrix .m16n16 / .m8n16 .b8` and `stmatrix .m16n8 .b8`; FP8 and `.acc::f16` for `multimem`; `fabric.*` (9.3). Two things that are often misreported as new: `ld.global.nc.L2::256B` has existed since sm_80, and FP4 / FP6 `mma.sync` exists only on sm_120.

## Software stack versions

| Component | Minimum for B200 | Notes |
| --- | --- | --- |
| CUDA | 12.8 | 12.9 adds the `f` suffix and `sm_103`; 13.0 renames `sm_101` → `sm_110`; current 13.3U1. DeepGEMM SM100 needs 12.9+ |
| Driver | 570.26 (12.8) / 580.65.06 (13.0) | |
| CUTLASS | 3.8 | 4.x recommended: 4.0 (June 2025) CuTe DSL; 4.2 SM103; 4.5 (May 2026) mixed-precision MXF8F6F4; 4.7.1 (August 2026) current stable |
| cuBLAS | 12.8 | block-scaled FP8 / FP4 from 12.8; cuBLASLt grouped GEMM from 13.1 |
| cuDNN | 9.7 | 9.13 FP8 SDPA; 9.21 SDPA with 2-CTA MMA and MXFP8 SDPA; current 9.25.1 |
| NCCL | 2.25.1 | GB200 MNNVL needs 2.25.2+ |
| NVSHMEM | 3.2.5 (NVLink 5 on B200) / 3.3.9 (NVL72 GA) | NVL72 needs Fabric Manager + IMEX |
| PyTorch | 2.7 (cu128) | the official wheel's arch list has no `a` suffix |
| Triton | 3.3 (tcgen05 + TMEM modelling); 3.4 Gluon | the old `tl._experimental_descriptor_*` API is gone in 3.3+ |

Other projects: FlashAttention goes through FA4 (see [`kernels/flashattention`](../kernels/flashattention.md)); ThunderKittens 2.0 (early 2026) supports SM100 / 103 / 120 with MXFP8 and NVFP4.

## Sources

- NVIDIA PTX ISA 9.3, "Tensor Core 5th Generation Instructions" and "Asynchronous Warpgroup Level Matrix Instructions"
- NVIDIA Blackwell Compatibility Guide; the compute-capability appendix of the CUDA C++ Programming Guide; the Blackwell Tuning Guide
- NVIDIA blog, "NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features"
- CUTLASS docs "Blackwell Functionality", "GEMM Performance Measurement Methodology Guidelines", the tcgen05 programming guide; `sm100_gemm_tma_warpspecialized.hpp`
- arXiv 2507.10789 (RTX 5080), 2512.02189 (B200 microbenchmarks), 2512.07004 (FP8 accumulation precision)
- DeepGEMM PR #112; the release notes of the libraries above

Numbers marked as third-party or unpublished (clocks, L2 partition count, TMEM versus occupancy) have no official NVIDIA source.
