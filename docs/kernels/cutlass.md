# CUTLASS

NVIDIA 的 CUDA 模板库。高性能 GEMM（现在也包括卷积及相关算子）的 C++ 模板参考实现。其他大多数 GPU kernel 库要么直接建立在 CUTLASS 之上，要么深受它的启发。

## 是什么

一个纯头文件的 C++ 模板库，模板实例化后编译成优化过的矩阵乘 CUDA kernel。用户实例化模板时指定：

- 操作数类型（BF16、FP8、NVFP4 等）
- tile 形状（例如 m128n128k64）
- 流水线级数
- 目标架构（例如 `cutlass::arch::Sm100`、`cutlass::arch::Sm120`）
- 布局（行主序/列主序）

然后 CUTLASS 为这个组合生成一个调优过的 kernel。现代的 CUTLASS（3.x）底层用 CUTE——一个高层的布局/代数库——来做清晰的索引计算。

GitHub：`NVIDIA/cutlass`。由 NVIDIA 维护。FlashInfer、vLLM（部分路径）、TensorRT-LLM、sglang、DeepSeek-AI 的技术栈等许多项目都用它作为 GEMM 后端。

## 依赖什么

- CUDA toolkit（SM100 需要 CUDA ≥ 12.8，CUTLASS 3.8 起支持）
- C++17
- 一个 nvcc 能驱动的 C++ 编译器

CUTLASS 本身在运行时不依赖任何其他东西——它是纯头文件，直接编进使用它的程序里。

## SM100 的情况

CUTLASS 3.8+ 在 `cutlass/include/cutlass/gemm/collective/sm100_*` 和 `cutlass/include/cutlass/gemm/kernel/sm100_*` 下有专门的数据中心版 Blackwell 模板。这些模板：

- 目标是 `sm_100a`（架构专用加速版）
- 内层 MMA 循环用 `tcgen05.mma`
- 用 `tcgen05.alloc` 在 TMEM 里分配累加器
- 默认用**单 CTA 模式**（`cta_group::1`）；CTA pair 模式（`cta_group::2`）通过选择 tile 形状来启用
- 为操作数暂存的流水线缓冲区申请最多约 220 KiB 的 SMEM
- 开启 CTA pair 模式时，依赖 cluster 共享的 TMA 做跨 CTA 数据搬运

用 `nvcc -gencode arch=compute_100,code=sm_100a` 编译它们，得到的 fatbin 只能在 SM 10.0 设备上运行。

## SM120 的情况

CUTLASS 3.6+ 在 `sm120_*` 下有一套平行的模板。这些模板：

- 目标是 `sm_120`（或者为了向前兼容用 `sm_120f`）
- 用 `mma.sync`（含 `sm_120a` 专属的块缩放 `mma.sync`）代替 `tcgen05.mma`
- 累加器放在**寄存器**里（tile 要小一些才装得下），或者经 SMEM 暂存（tile 更大，但 SMEM 压力也更大）
- **只用单 CTA**（没有 `cluster_dim > 1`）
- 限制流水线级数，以适应 99 KiB 的 SMEM 上限
- 达到 SM120 最优吞吐的约 40–70 %，作为对比，SM100 模板在 SM100 上能达到约 95 %

SM120 模板和 SM100 模板是*两棵独立的树*，不是简单地重新编译一遍。

## SMEM 断崖

工作站 Blackwell 上最常撞见的 CUTLASS 问题。来龙去脉：

1. CUTLASS 用 `StageCountAutoCarveout<sizeof(SharedStorage)>` 来决定在可用 SMEM 里能塞下多少级流水线。
2. `StageCountAutoCarveout` 按 `total_smem - other_uses` 算剩余 SMEM，其中 `total_smem` 取自该架构公布的最大值。
3. SM100 上最大值是 228 KiB。SM120 上最大值是 **99 KiB**。
4. 如果开发者在 SM100 上（228 KiB 的余量）测试了自己的模板，然后把同一份代码拿到 SM120 上跑，自动划分的计算会以为有 228 KiB 可用，申请的流水线缓冲区就会超出实际的 99 KiB。
5. `cudaFuncSetAttribute` 申请超过 99 KiB 会返回 `cudaErrorInvalidValue`，直接启动会报 out of resources。错误只在运行时出现，编译期没有任何提示，而且上层库常把它吞掉、换成一句含糊的"kernel 不可用"。

标志性的 issue：`NVIDIA/cutlass#3096`（"SMEM size detection on Blackwell consumer parts"）。正在推进的修复：在运行时查询 SMEM 预算，尊重设备的实际上限。

在此之前的临时办法：

- 手动把 `StageCount` 设成一个小数字（2 或 3），不用自动划分
- 用 `sm120_*` 模板而不是 `sm100_*` 模板
- 选更小的 tile 形状，别把 SMEM 用得太满

## CUTLASS 的 SM100 内核长什么样

从 Hopper 的 `sm90_*` 模板过来，命名和分工都变了：

- kernel schedule：`KernelTmaWarpSpecialized1SmSm100` / `2SmSm100`；块缩放的 `...{1,2}SmBlockScaledSm100`，再细分 `Nvf4 / Mxf4 / Mxf8f6f4`。`KernelScheduleAuto` 按 cluster 形状自动选 1SM 还是 2SM
- mainloop：`MainloopSm100TmaUmmaWarpSpecialized`
- epilogue：`Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>`，从 TMEM 读累加器用 `SM100_TMEM_LOAD_16dp256b1x / 32dp32b1x` 这类拷贝原子
- 参考 warp 分工：warp 0 = MMA（一个 lane 发射）、warp 1 = 调度（CLC）、warp 2 = TMA load、warp 3 = epilogue load、warp 4 起 = epilogue（默认 4 个 warp，各读自己那 1/4 lane）。Hopper 的"2 个 consumer warpgroup 各持一份累加器寄存器"模型作废：MMA 一个线程、TMA 一个线程，照搬会浪费 warp 和寄存器
- 版本：3.8 首支持 SM100；4.0（2025-06）加 CuTe DSL；4.2 加 SM103；4.5（2026-05）加 MXF8F6F4 混精；截至 2026 年 9 月稳定版 4.7.1

源码入口：`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`。

## 常见故障

**故障 1：`no kernel image is available`**

你把一个用了 CUTLASS 的库按 `sm_100a` 编译，然后在 SM 12.0 上跑。fatbin 里只有 `sm_100a` 的 cubin，没有 `sm_120` 的 SASS，内嵌的 PTX（如果有的话）也是面向 `sm_100a` 的，无法 JIT 到 `sm_120`。

修复：用 `-gencode arch=compute_120,code=sm_120` *并且*换用面向 SM120 的模板重新编译，而不是只给 SM100 模板换个 gencode 参数。

**故障 2：SMEM 超限，启动失败（SMEM 断崖）**

上面已经讲过。症状是 `cudaErrorInvalidValue` 或 launch out of resources，常被上层库包装成"kernel 不可用"。

检测：`nvcc --ptxas-options=-v` 看该模板实例的 SMEM 用量是否超过 99 KiB；或者在启动前后检查 `cudaGetLastError`。

修复：用 SM120 模板配上更小的 tile 形状，或者显式设置 `StageCount`。

**故障 3：CTA pair MMA**

一个带 `cta_group::2`（CTA pair MMA）的 CUTLASS 模板为 SM120 编译。`ptxas` 会在编译期拒绝 `tcgen05.*`；如果是预编译的 `sm_100a` cubin，加载时就报没有可用的 kernel image。

修复：只用带 `cta_group::1` 的 CUTLASS 模板。SM120 模板树强制了这一点；SM100 模板树没有。

**故障 4：NVFP4 缩放因子布局不匹配**

CUTLASS 要求 NVFP4 的缩放因子按特定布局存放（块交错，FP8 E4M3）。如果模型文件是按 MX-FP4 布局保存的（块大小 32，缩放因子是 E8M0），再加载进 CUTLASS 的 NVFP4 模板，缩放因子就会被错误解读。

修复：重新量化模型文件，或者换一个布局匹配的 kernel 库。

## 检测方法

检查一个 `.so` 是否用了 CUTLASS：

```bash
nm -D mylib.so | grep -i cutlass | head
# 或者
strings mylib.so | grep -E 'cutlass::|CollectiveBuilder|StageCount' | head
```

检查里面有哪些架构目标：

```bash
cuobjdump --list-elf mylib.so
```

看有没有 `arch = sm_100a`（只能跑在数据中心版）或 `arch = sm_120`（工作站可用）。

## 阅读 CUTLASS 源码

这个库很大（约 20 万行代码），但组织得很清楚：

```
include/cutlass/
├── gemm/
│  ├── collective/
│  │  ├── sm70_*    # Volta
│  │  ├── sm80_*    # Ampere
│  │  ├── sm90_*    # Hopper
│  │  ├── sm100_*   # 数据中心版 Blackwell
│  │  └── sm120_*   # 工作站 Blackwell
│  ├── kernel/     # 顶层 kernel 组装
│  └── threadblock/   # 旧的（3.x 之前）tile 级代码
├── arch/        # 架构封装（cutlass::arch::Sm120 等）
├── conv/        # 卷积（结构类似）
└── ...
```

想搞清楚哪些东西是某个架构特有的，把 `sm100_*` 和 `sm120_*` diff 一下。差异会集中在 MMA 指令封装、tile 形状和流水线深度上。

## 值得关注的 CUTLASS issue

几个能反映 SM120 现状的、开放的或近期的 issue：

- `#3096` — 消费级 Blackwell 上的 SMEM 大小检测
- `#3045` — NVFP4 缩放因子布局不一致
- `#2950` — sm120 模板的 stagecount 自动划分
- `#3120` — sm120 的 wgmma 回退路径

这些 issue 是 CUTLASS 开发的当前前沿；它们的解决会影响后续版本里 SM120 上什么能用。

## 另见

- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) — `mma.sync`、`wgmma`、`tcgen05`
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — `sm100_*` 模板用的东西
- [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md) — SMEM 断崖
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) — 移植套路
- GitHub 上的 *NVIDIA/cutlass*
- *CUTLASS Programming Guide*（在仓库的 `media/docs/` 目录下）
