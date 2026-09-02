# Marlin 及其同类——INT4 GEMM 路径

老一些但仍然有用：INT4 量化配 FP16（或 BF16）激活。早于 Blackwell 出现，在工作站 Blackwell 上工作正常。对很多部署来说，这是阻力最小的路。

## Marlin

Marlin 是一个实现 **W4A16 GEMM** 的 CUDA kernel——INT4 权重、FP16 激活、FP16 或 FP32 累加器。针对 Ampere（SM80）调优，一路兼容到 Blackwell。

GitHub：`IST-DASLab/marlin`。协议：Apache-2.0。

### 它做什么

一种特殊的 GEMM：

- 权重是 **4 位整数**，外加单独的缩放因子（通常按输出通道或按组）
- 激活是 FP16
- Tensor Core MMA 以 FP16 进行；权重在计算时用每组的缩放因子即时反量化
- 输出是 FP16 累加器（可选 FP32）

小 batch（decode 型负载）下的吞吐能和 FP8 GEMM 打平，而**权重占用的内存带宽只有一半**。

### SM 兼容性

| 架构 | 支持情况 |
| --- | --- |
| Ampere（SM 8.0/8.6/8.9） | 完整支持，设计目标 |
| Hopper（SM 9.0） | 可用，比 wgmma-FP8 路径慢约 10–20 % |
| 数据中心版 Blackwell（SM 10.0） | 可用，比原生 NVFP4 慢 |
| 工作站版 Blackwell（SM 12.0） | **可用**，往往是实际可行的快速路径 |

在工作站 Blackwell 上，当 NVFP4 路径走不通或有 bug 时（DeepGEMM 没移植、CUTLASS NVFP4 撞上 SMEM 断崖），Marlin 是一个可行的替代。代价是权重略大一点（含元数据约 4 位 → 约 4.25 位，对比 NVFP4 的 4.5 位——其实反而略*小*），以及在某些模型上精度略低。

### 常见故障

- **组大小不匹配**：Marlin 对量化时用的组大小（通常是 128 或 64）很敏感。用一个组大小量化的模型，交给期望另一个组大小的 kernel 加载，输出就是错的。
- **激活精度不匹配**：把 BF16 激活喂给期望 FP16 的 Marlin kernel，输出会满是 NaN（FP16 最大值约 65 504，BF16 的范围更宽）。

## AWQ——激活感知的权重量化

这是一种*量化方案*，不是 kernel——但它附带 kernel。AWQ 根据激活的幅度来决定哪些权重列可以激进地量化、哪些要保留在 FP16。经验上，同样的比特率下，精度比朴素的 INT4 量化更好。

AWQ 的 kernel 用的技术和 Marlin（W4A16）类似，只是带上了激活感知量化的元数据。

GitHub：`mit-han-lab/llm-awq`。支持 SM80–SM120。

## GPTQ——基于梯度的训练后量化

另一种量化方案。利用校准数据集上的二阶梯度信息来优化每个权重的舍入。通常比朴素的 RTN（就近舍入）效果更好，但计算更慢。

GPTQ 格式是很多 kernel 的目标格式；Marlin 和 ExLlama（另一个推理库）都有 GPTQ 格式的读取器。

GitHub：`IST-DASLab/gptq` 及其下游。支持 SM80–SM120。

## 它们在整个栈里的位置

对工作站 Blackwell 用户来说，部署大模型的现实选项有：

1. **NVFP4 + CUTLASS**：能跑起来的时候吞吐最好。部分 kernel 会撞上 SMEM 断崖。
2. **NVFP4 + DeepGEMM**：尚未移植到 SM120（截至 2026 年初）。
3. **NVFP4 + FlashInfer**：不涉及 MoE all-to-all 的路径可用。
4. **W4A16 + Marlin/AWQ/GPTQ**：吞吐低于 NVFP4（Tensor Core 跑在 FP16 峰值而非 FP4 峰值），但稳定可靠。
5. **FP8 + CUTLASS**：权重内存是 NVFP4 的 2 倍，但没有 SMEM 断崖，kernel 支持也更广。

选哪个往往由模型有什么格式决定——如果一个模型只发布了 NVFP4 版本，你没法轻易转成 W4A16。反过来，社区量化的模型很多是 GPTQ/AWQ 格式，和 Marlin 配合得很好。

## 其他值得了解的格式

- **EXL2 / EXL3**（ExLlama）：社区流行的自定义格式，逐层混合精度
- **bitsandbytes（NF4）**：集成在 HuggingFace 里，LoRA 训练常用
- **OpenCompute MX-INT4 / MX-INT8**：与 MX-FP4 类似的结构化格式

在工作站 Blackwell 上做生产推理，**通过 Marlin 跑 GPTQ 格式的 W4A16** 是最简单的可行路径。不是最快的，但能用。

## 另见

- [`fundamentals/number-formats`](../fundamentals/number-formats.md)——更全面的格式全景
- [`blackwell/nvfp4-deep-dive`](../blackwell/nvfp4-deep-dive.md)——NVFP4 这个替代选项
- *Frantar et al., "GPTQ"*（2023）
- *Lin et al., "AWQ"*（2024）
- GitHub 上的 `IST-DASLab/marlin`
