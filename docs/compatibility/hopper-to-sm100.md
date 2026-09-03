# 从 Hopper 迁到 B200

本 wiki 的其他页讲的是"SM100 的东西怎么搬到 SM120"。这一页反过来：手里有一套在 H100（`sm_90a`）上跑得好好的 kernel，要在 B200（`sm_100a`）上跑，会踩哪些坑。按从"编不过"到"跑得慢"的顺序排。事实来源见文末。

## 1. 二进制层：什么都带不过去

- `wgmma` 一行都带不过去，也没有降级路径。`wgmma.mma_async / fence / commit_group / wait_group` 在 PTX 里都只有 "Requires sm_90a"。
- `sm_90a` 的 cubin 和 PTX 都加载不了；`sm_90` 的 cubin 也不行（9 → 10 换了主版本）；只有不带 a 的 PTX 能 JIT。检查手段：`CUDA_FORCE_PTX_JIT=1`。
- 编译目标写 `sm_100` 等于没有 `tcgen05`。要 `sm_100a`；想同时覆盖 B300 就 `sm_100f`。PyTorch 扩展要显式写 `10.0a`。
- 最低 CUDA 12.8、驱动 570.26；`f` 后缀要 12.9。CUDA 13.0 删了 `cudaDeviceProp.clockRate` 等字段，老 harness 编不过。
- `mma.sync` 是唯一不重写就能跑的 MMA 路径，但 B200 上 FP8 走它是转成 FP16 的 HMMA。

细节见 [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md)。

## 2. 别拿 RTX 50 的数据当 B200

arXiv 2507.10789 测的是 RTX 5080（SM120），它的 L2、SMEM、`mma.sync` 数据都不适用于 B200；B200 的微基准是 arXiv 2512.02189。数据表上的算力全是稀疏值，稠密除 2；HGX B200 和 GB200 里的 GPU 是两档。见 [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md)。

## 3. TMEM 是运行时资源，静态 occupancy 算不出来

- `tcgen05.alloc` 资源不够时**阻塞**而不是失败；同一 SM 上第二个 CTA 要 512 列会永远等。
- alloc 之后立刻 `relinquish_alloc_permit`；后一次 alloc 不得比前一次大；退出前必须 `dealloc`。
- `cudaOccupancy*` 不知道 TMEM 的存在。每 SM 能驻留几个 CTA，由你分配的列数在运行时决定。

见 [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md)。

## 4. warp 分工要重排

Hopper 的"1 个 producer warpgroup + 2 个 consumer warpgroup 各持累加器寄存器"模型作废：

- `tcgen05.mma` 由**一个线程**发射，TMA 也是一个线程。照搬 Hopper 的分工会浪费 warp 和寄存器。
- epilogue 至少 4 个 warp，每个只能读自己那 32 个 lane 的 TMEM；一个 warp 读不到整块累加器。
- 完成机制分两套：`tcgen05.ld / st` 用 `wait::ld / st`，`mma / cp / shift` 用 `commit` 加 mbarrier。不能混。
- 整个 kernel 的 `.cta_group` 必须一致。
- CUTLASS 的参考分工：warp 0 MMA、warp 1 调度、warp 2 TMA、warp 3 epilogue load、warp 4 起 epilogue。见 [`kernels/cutlass`](../kernels/cutlass.md)。

## 5. 形状和数据类型的坑

- 块缩放 kind（`mxf8f6f4 / mxf4 / mxf4nvf4`）没有 M=64，只有 M=128。
- `mxf4 / mxf4nvf4 / i8` 是 `sm_100a` 专属，`sm_100f` 没有。
- 缩放因子要先 `tcgen05.cp` 进 TMEM，CUTLASS 规定的 GMEM 布局固定且不能转置。
- `kind::f8f6f4` 里 4/6 位元素各占一个字节，只有 `mxf4` 系才真正打包。
- INT4 在 `tcgen05` 里不存在。
- 见 [`fundamentals/number-formats`](../fundamentals/number-formats.md)。

## 6. FP8 不用再 promotion

Hopper 上每 128 个 K 回 CUDA core 累加一次的做法，在 B200 上是白付的：`tcgen05` 的 FP8 累加精度实测 25 个小数位。但缩放因子格式要换成打包的 UE8M0。DeepGEMM 的 SM100 内核就是这么改的，见 [`kernels/deepgemm`](../kernels/deepgemm.md)。

## 7. 2-SM MMA 的分工

- B 是被共享的那个：每个 CTA 只装 N 一半的 B；A 和 D 各自私有。
- cluster 的 CTA 总数必须是偶数；只有 leader 发 MMA。
- TMA 完成信号用 `.cta_group::2` 打到对方 CTA 的 mbarrier；`tcgen05.commit` 要 multicast 到两边。
- 见 [`blackwell/thread-block-clusters`](../blackwell/thread-block-clusters.md)。

## 8. 频率、功耗和测量

- NVIDIA 没有公布 B200 的频率。从额定算力反推 HGX B200 约 1.9 GHz、GB200 约 2.1 GHz；nvidia-smi 上有人记到上限 1.965 GHz、base 700 MHz。这些都不是官方数字。
- 到功耗帽就 DVFS，锁频也压不住。CUTLASS 的性能测量指南建议：数据 zero-fill、监控频率、长 warmup 加 cool-down（Blackwell 大 GEMM 用 10000 次 warmup、4000 次迭代、1 秒 cool-down）。
- HGX B200 可配到 1000 W，GB200 到 1200 W，DGX B200 默认帽 700 W。

## 9. L2 分区

126 MB L2 分成多个分区，跨分区带宽从约 21 TB/s 降到约 17 TB/s、延迟明显升高。分区数第三方说法不一（2 或 4），NVIDIA 没有文档，也没有 SM 到分区的映射 API。见 [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md)。

## 10. 工具

- ncu 从 2024.4 起支持 Blackwell。Profiling Guide 定义了 `tc`（UTCBAR / UTCCP / UTC\*MMA / UTCSHIFT）、`tmem`（LDT / STT）、`tma` 三组 pipeline。
- 解压引擎（Snappy / LZ4 / Deflate / Gzip，最高 600 GB/s）只有 host 侧 driver API `cuMemBatchDecompressAsync`，没有 device 端 PTX 接口。和 `tcgen05.cp` 的 FP4 / FP6 解包是两回事。
- 其它 `sm_100` 新 PTX：`redux.sync.f32`；256 位 `ld/st .v8.b32` 和 `.level2::eviction_priority`（8.8）；`ldmatrix .m16n16 / .m8n16 .b8` 与 `stmatrix .m16n8 .b8`；`multimem` 的 FP8 和 `.acc::f16`；`fabric.*`（9.3）。两个误传：`ld.global.nc.L2::256B` 不是新的（sm_80 就有）；`mma.sync` 的 FP4 / FP6 只在 sm_120。

## 软件栈版本线

| 组件 | B200 最低 | 备注 |
| --- | --- | --- |
| CUDA | 12.8 | 12.9 加 `f` 后缀与 `sm_103`；13.0 改名 `sm_101` → `sm_110`；当前 13.3U1。DeepGEMM SM100 要 12.9+ |
| 驱动 | 570.26（12.8）/ 580.65.06（13.0） | |
| CUTLASS | 3.8 | 建议 4.x：4.0（2025-06）CuTe DSL；4.2 SM103；4.5（2026-05）MXF8F6F4 混精；4.7.1（2026-08）当前稳定 |
| cuBLAS | 12.8 | 块缩放 FP8 / FP4 从 12.8；cuBLASLt grouped GEMM 13.1 起 |
| cuDNN | 9.7 | 9.13 FP8 SDPA；9.21 SDPA 2-CTA MMA、MXFP8 SDPA；当前 9.25.1 |
| NCCL | 2.25.1 | GB200 MNNVL 要 2.25.2+ |
| NVSHMEM | 3.2.5（B200 NVLink 5）/ 3.3.9（NVL72 GA） | NVL72 需 Fabric Manager + IMEX |
| PyTorch | 2.7（cu128） | 官方 wheel 的 arch list 不带 a |
| Triton | 3.3（tcgen05 + TMEM 建模）；3.4 Gluon | 旧 `tl._experimental_descriptor_*` 在 3.3+ 删除 |

其它项目：FlashAttention 走 FA4（见 [`kernels/flashattention`](../kernels/flashattention.md)）；ThunderKittens 2.0（2026 年初）支持 SM100 / 103 / 120，含 MXFP8 和 NVFP4。

## 来源

- NVIDIA PTX ISA 9.3，"Tensor Core 5th Generation Instructions"、"Asynchronous Warpgroup Level Matrix Instructions"
- NVIDIA Blackwell Compatibility Guide；CUDA C++ Programming Guide 的计算能力附录；Blackwell Tuning Guide
- NVIDIA 博客 "NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features"
- CUTLASS 文档 "Blackwell Functionality"、"GEMM Performance Measurement Methodology Guidelines"、tcgen05 编程指南；`sm100_gemm_tma_warpspecialized.hpp`
- arXiv 2507.10789（RTX 5080）、2512.02189（B200 微基准）、2512.07004（FP8 累加精度）
- DeepGEMM PR #112；各库的 release notes

标了"第三方"或"未公布"的数字（频率、L2 分区数、TMEM 与 occupancy 的关系）没有 NVIDIA 官方来源。
