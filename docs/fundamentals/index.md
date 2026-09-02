# 基础

要读懂 wiki 其余部分所需的最少背景。如果你写过 CUDA kernel、看 PTX 清单也不费劲，快速扫一遍这几页就往下走。如果没有，请仔细读——后面的内容默认你已经掌握了这里建立的词汇。

## 本节页面

- [`gpu-execution-model`](gpu-execution-model.md) — warp、block、SM、grid 层级，SIMT 抽象
- [`memory-hierarchy`](memory-hierarchy.md) — 寄存器、SMEM、L1/L2、全局内存，带宽金字塔
- [`cuda-pipeline`](cuda-pipeline.md) — `nvcc` → PTX → `ptxas` → cubin → 驱动 → SASS
- [`tensor-cores`](tensor-cores.md) — Tensor Core 是什么、各代演进，以及 Blackwell 之前的 `mma.sync` 和 `wgmma.async`
- [`number-formats`](number-formats.md) — FP16/BF16、FP8（E4M3/E5M2）、FP6、FP4，以及 MX-FP4/NVFP4 家族

## 这里不讲什么

- **CUDA C++ 语言细节。** 本 wiki 只把 CUDA C++ 当作编译流水线的输入，不教语言本身。NVIDIA 的 *CUDA C++ Programming Guide* 是权威来源。
- **通用 GPU 编程模式。** 分块、占用率、寄存器压力——这些会在后面章节通过 Blackwell 相关的例子间接讲到。
- **性能工具。** Nsight Systems、Nsight Compute。在 [`overview/getting-started`](../overview/getting-started.md) 里顺带提到，这里不教。
- **Volta 之前的历史。** 本 wiki 从 Volta 讲起，因为从那一代开始 Tensor Core 成为一等概念，`mma.sync` 也进入了 ISA。

## 为什么要有这一节

下一节的 Blackwell 专属内容，只有在你知道 SM100/SM120 分裂中哪些东西*没有*变的前提下才说得通。Tensor Core 仍然执行 MMA。共享内存仍然分 bank、仍然有 bank 冲突。CUDA 编译流水线仍然先出 PTX 再出 SASS。变的是*哪些* PTX 指令存在、共享内存有*多少*可用、*哪些* tile 形状能跑——如果你不知道被改的东西原本是什么样，这些变化就都看不见。

这几页建立的是基线。后面的章节描述的是相对这条基线的差异。

## 建议顺序

1. [`gpu-execution-model`](gpu-execution-model.md) — 抽象机器
2. [`memory-hierarchy`](memory-hierarchy.md) — 数据放在哪
3. [`tensor-cores`](tensor-cores.md) — 矩阵乘硬件做什么
4. [`number-formats`](number-formats.md) — 矩阵乘硬件*操作的是什么*
5. [`cuda-pipeline`](cuda-pipeline.md) — 源码怎样变成硬件指令

大多数读者会觉得第 1–4 页是复习，而第 5 页需要仔细读。如果你也是这样，直接跳到 [`cuda-pipeline`](cuda-pipeline.md)。
