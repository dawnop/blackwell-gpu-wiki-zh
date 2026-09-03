# 参考文献

精选的延伸阅读资料，按主题分组。

## NVIDIA 架构与 PTX

- **NVIDIA. *Blackwell Architecture Whitepaper*.** 2024–2025。SM100 和 SM120 设计的权威参考。
- **NVIDIA. *PTX ISA Reference Manual*.**（原文引用 8.5；`tcgen05` 自 8.6 / CUDA 12.8 起，截至 2026 年 9 月为 9.3 / CUDA 13.3）与 CUDA 12.x / 13.x 同步。包含 `mma.sync`、`tcgen05.*`、`cp.async`、`cp.async.bulk`（TMA）、线程块簇、分布式共享内存的规范。
- **NVIDIA. *CUDA C++ Programming Guide* (current edition).** 关于 Tensor Core、异步拷贝、cooperative groups、cluster 启动的章节。
- **NVIDIA. *CUDA C++ Best Practices Guide*.** 内存层次的使用建议、占用率调优。

## 数值格式

- **NVIDIA. *NVFP4 Block Format Specification*.** 属于 CUTLASS 文档的一部分。定义了 16 元素的块大小、每块一个 FP8（E4M3）缩放因子，以及 SM100 与 SM120 各自的布局。
- **OCP (Open Compute Project). *Microscaling (MX) Formats Specification*, v1.0**. 2023 年 10 月。定义了 MX-FP4、MX-FP6、MX-FP8（NVFP4 的开放格式对应物）。
- **NVIDIA. *FP8 Formats for Deep Learning*.** 白皮书，2022 年。E4M3 和 E5M2 的细节、缩放方式及对精度的影响。

## Tensor Core 与矩阵运算

- **CUTLASS GitHub repository.** [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) —— 理解 Tensor Core 调用方式的首要来源。`include/cutlass/arch/mma_sm100.h` 和 `mma_sm120.h` 两个文件以代码形式记录了 `tcgen05` 和 `mma.sync`。
- **NVIDIA. *Programming Tensor Cores in CUDA*.** 开发者博客文章，有多个版本。

## FlashAttention

- **Dao, Tri, et al. "*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*."** NeurIPS 2022。
- **Dao, Tri. "*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*."** 2023。
- **Shah, Jay, et al. "*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*."** 2024。Hopper 专属设计；记录了 warp 特化的两级流水线，它无法干净地搬到 SM100 上，更不用说 SM120。
- **FlashAttention GitHub repo:** [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。FA2 源码可移植；FA3 仅限 Hopper。

## DeepSeek 架构

- **DeepSeek-AI. *DeepSeek-V3 Technical Report*.** 2024 年 12 月。引入了 MLA（Multi-Latent Attention）。
- **DeepSeek-AI. *DeepSeek-V3.2 Technical Report*.** 引入了 DSA（DeepSeek Sparse Attention）。
- **DeepSeek-AI. *Native Sparse Attention*.** 提出 NSA 的论文，外界普遍猜测 V4 会采用。
- **DeepGEMM GitHub repository:** [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。FP8/FP4 GEMM kernel，面向 SM100。

## Kimi K2

- **Moonshot AI. *Kimi K2 Technical Report*.** 2024。K2 系列的背景、MuonClip 优化器、MoE 结构。
- **Kimi K2 model card on Hugging Face**：超参数和架构细节。

## GLM-5

- **Zhipu AI / Tsinghua KEG. *GLM-5.1 Technical Report* (forthcoming or released depending on version).** GLM-5.1 478B-A42B 的架构细节。
- **REAP Pruning paper.** 按路由权重的专家剪枝方法，REAP-160 和 REAP-172 由此而来。

## NCCL 与集合通信

- **NVIDIA. *NCCL User Guide*.** 记录了 `NCCL_P2P_LEVEL`、`NCCL_IB_DISABLE`、ring 与 tree 算法、各环境变量。
- **NCCL GitHub repository:** [github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)。理解 ring all-reduce 机制的源码。

## NVSHMEM 与 PGAS

- **NVIDIA. *NVSHMEM Documentation*.** 面向 GPU 集群的 PGAS 抽象。
- **NVSHMEM GitHub repository:** [github.com/NVIDIA/nvshmem](https://github.com/NVIDIA/nvshmem)（或通过开源镜像获取）。

## DeepEP

- **DeepEP GitHub repository:** [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)。MoE 专家并行通信库，面向数据中心版 Blackwell。

## 推理引擎

- **vLLM GitHub repository:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)。PagedAttention、连续批处理。
- **sglang GitHub repository:** [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)。面向结构化输出的推理引擎。
- **TensorRT-LLM GitHub repository:** [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)。NVIDIA 的生产级推理编译器。
- **FlashInfer GitHub repository:** [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)。各主流引擎都在用的注意力与 MoE kernel 库。

## MoE all-to-all

- **PPLX library**：另一种 MoE all-to-all 实现，sglang 和 vLLM 有时用它作为不依赖 NVSHMEM 的路径。

## Marlin 与仅权重量化

- **Frantar, Elias, et al. "*Marlin: Fast 4-bit Inference Kernels*."** 2024。混合精度 GEMM 设计。
- **Marlin GitHub repository.**

## Triton

- **Tillet, Philippe, et al. "*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*."** MAPL 2019。
- **Triton GitHub repository:** [github.com/triton-lang/triton](https://github.com/triton-lang/triton)。

## 混合专家的奠基工作

- **Shazeer, Noam, et al. "*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*."** ICLR 2017。现代 transformer MoE 的起点。
- **Fedus, William, et al. "*Switch Transformers*."** JMLR 2022。
- **Lepikhin, Dmitry, et al. "*GShard*."** ICLR 2021。

## REAP 剪枝

- **(REAP paper, 2024–2025).** "Router-weighted Expert Activation Pruning" —— 描述了 REAP-160 / REAP-172 的方法。

---

这个 wiki 本身是带观点的综合整理，不是一手资料。有疑问时，以 NVIDIA 文档和 kernel 库源码为准。
