# 术语表

本项目专用和 Blackwell 专属的词汇。每个词条都链接到深入讲解它的页面。

## 计算能力

**Compute capability（计算能力，CC）**——NVIDIA 为 SM ISA 定的版本编号方案，写作 `<主版本>.<次版本>`（例如 `7.0` Volta、`8.0` Ampere、`9.0` Hopper、`10.0` 数据中心版 Blackwell、`12.0` 工作站/消费级 Blackwell）。主版本相同通常意味着 PTX 向前兼容；主版本不同则某些特性可能根本不存在。见 [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md)。

**`sm_NN`**——计算能力的编译器标志小写形式。`sm_100`、`sm_120`。不带后缀的形式：该架构的可移植子集。

**`sm_NNa`**——"架构专属加速"（architecture-specific accelerated）。允许使用不可移植的指令（例如 `sm_100a` 开启 `tcgen05.*`）。带 `a` 后缀编译出的代码只能在这一个精确的计算能力上运行——早的不行，晚的也不行。

**`sm_NNf`**——"家族专用"（family-specific）。把代码限制在能在 `sm_NN` 以及同一家族后续架构上运行的指令范围内。适合需要同时在 `sm_120` 工作站芯片和未来任何 `sm_12N` 芯片上运行的代码。

## 架构与代号

**Blackwell**——NVIDIA 的 GPU 世代，2024–2026 年。
- **GB100**：B100/B200 数据中心芯片，SM 10.0，HBM3e，NVLink 5。
- **GB202**：工作站/消费级芯片（RTX PRO 6000 Workstation、RTX 5090），SM 12.0，GDDR7，无 NVLink。

**Hopper**——NVIDIA 的上一代。SM 9.0（H100/H200）。引入了 TMA、异步 Tensor Core（`wgmma.async`）、线程块簇。

**Ampere**——Hopper 的上一代。SM 8.0/8.6/8.9（A100、RTX 30/40 系列）。

**SXM**——NVIDIA 数据中心卡的板型。有 SXM 就意味着有 NVLink。（RTX 卡是 PCIe 板型；"PRO"产品线两种都有。）

## 内存

**Global memory（全局内存，HBM/GDDR）**——片外显存。数据中心版 Blackwell 用 HBM3e（约 3–8 TB/s），工作站用 GDDR7（约 1.6 TB/s）。

**Shared memory（共享内存，SMEM）**——片上、每 block 独享的暂存区，由程序员管理。容量：SM120 上每 block 99 KiB，SM100 上每 block 228 KiB。99 与 228 之分是本 wiki 里影响最大的数字之一。见 [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md)。

**Tensor Memory（张量内存，TMEM）**——SM100 新引入的一类片上内存。存放 Tensor Core 的累加器，与寄存器解耦。**SM120 上不存在。** 见 [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md)。

**Registers（寄存器）**——每线程私有的存储。上限：目前大多数架构上每线程 255 个 32 位寄存器。

**Constant memory（常量内存）**——小容量、带缓存、只读的一类内存。

**L1 / L2 cache（L1 / L2 缓存）**——标准内存层级中的片上缓存。

## Tensor Core ISA

**`mma.sync`**——通用的 Tensor Core MMA 指令，自 Volta 起可用。同步：warp 阻塞直到结果落入寄存器。作用于小 tile（m16n8k16 / m16n8k32）。SM100 和 SM120 上**都**可用。

**`wgmma.async`**——Hopper 的 warp 组异步 MMA。tile 更大，异步执行。只在 `sm_90a` 上可用：数据中心版 Blackwell 换成了 `tcgen05.mma`，工作站版 Blackwell 只有 `mma.sync`。

**`tcgen05.mma`**——数据中心版 Blackwell 的 MMA 指令族。异步、大 tile（单 CTA 最大 M=128、N=256，CTA pair 最大 M=256、N=256；K 按数据宽度定，FP4 时为 64），累加器放在 TMEM 里。**仅数据中心版可用。** 见 [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md)。

**`tcgen05.alloc` / `tcgen05.commit` / `tcgen05.ld` / `tcgen05.st` / `tcgen05.cp`**——配套指令：分配 TMEM、把已发射的异步操作挂到 mbarrier 上等完成、TMEM 与寄存器之间读写、从 SMEM 拷进 TMEM（只有这一个方向，TMEM 里的结果要经寄存器才能出来）。

## 数值格式

**FP16**——IEEE 半精度浮点。1 位符号 + 5 位指数 + 10 位尾数。

**BF16**——brain-float-16。1 位符号 + 8 位指数 + 7 位尾数。范围与 FP32 相同，精度更低。

**FP8 E4M3**——8 位浮点，4 位指数、3 位尾数。精度较高，范围较小。

**FP8 E5M2**——8 位浮点，5 位指数、2 位尾数。范围较大，精度较低。训练中常用于梯度。

**FP6**——6 位浮点（E2M3 或 E3M2）。不太常见；出现在一些量化方案里。

**FP4（E2M1）**——4 位浮点，1 位符号 + 2 位指数 + 1 位尾数。范围极小；只有以**块量化**形式、配上每块一个缩放因子才有用。

**MX-FP4**——Open Compute Project 的微缩放（Microscaling）FP4 规范。32 个元素一块，每块一个 E8M0（8 位、只有指数）缩放因子。

**NVFP4**——NVIDIA 版的 MX-FP4。**16 个元素一块**（块更小，动态范围跟得更紧），每块一个 **FP8（E4M3）缩放因子**（比 E8M0 的缩放精度更高）。SM100 和 SM120 的 Tensor Core 都原生支持。见 [`fundamentals/number-formats`](../fundamentals/number-formats.md)。

**TF32**——Ampere 及之后 Tensor Core 内部使用的 19 位格式。1 位符号 + 8 位指数 + 10 位尾数。用于让 FP32 矩阵乘走 Tensor Core 加速。

## 并行方案

**TP（Tensor Parallelism，张量并行）**——把每个权重矩阵切到 N 张 GPU 上，每层做一次 all-reduce。适合 GEMM 密集型模型，对拓扑没有要求。

**PP（Pipeline Parallelism，流水线并行）**——把各层分到 N 张 GPU 上，以微批次流过。适合访存受限的模型。有一定的气泡开销。

**EP（Expert Parallelism，专家并行）**——用于 MoE：每张 GPU 拥有一部分专家，token 通过 all-to-all 路由过去。极耗带宽；只在 NVLink 级别的 fabric 上表现良好。见 [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md)。

**DP（Data Parallelism，数据并行）**——复制整个模型，切分 batch。训练中常见，推理中较少。

**Hybrid plans（混合方案）**——TP + PP、EP + TP 等。大多数机器上的务实选择。

## 互连

**NVLink**——NVIDIA 私有的高带宽 GPU 间互连。第 5 代（Blackwell）：NVL72 上每 GPU 1.8 TB/s。**仅数据中心版可用。**

**NVSwitch**——基于 NVLink 的交换 fabric。在单个机箱（DGX、HGX）内连接 8 张以上 GPU，任意两卡之间带宽一致。

**MNNVL（Multi-Node NVLink，多节点 NVLink）**——NVL72 级别的 fabric，把 NVLink 扩展到跨机架（最多 72 张 GPU）。

**PCIe**——通用的主机侧互连。Gen4 每 lane 16 GT/s（约 2 GB/s 每方向），Gen5 每 lane 32 GT/s（约 4 GB/s 每方向）。x16 → 每方向约 32 GB/s（Gen4）或 64 GB/s（Gen5）。

**P2P（peer-to-peer）**——GPU 之间直接访问彼此显存，不经过主机内存中转。GPU 共用同一个交换芯片或 root complex 时可用。

**Atomics（原子操作）**——跨互连的原子内存操作。消费级 GPU 上，P2P 原子操作默认被软件锁住；需要 BIOS（ACS Disabled）+ 驱动（`RMDisableFeatureDisablement=1`）两处设置才能打开。见 [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md)。

**ACS（Access Control Services）**——PCIe 的一项特性，**开启**时把设备隔离到各自的 IOMMU 组里，并**阻断** P2P 原子操作。有点反直觉：要允许原子操作，得把 ACS 关掉。

**RDMA**——经网络（InfiniBand、RoCE）的远程 DMA。数据中心里用于多节点 GPU 间传输。与单节点消费级环境无关。

## 内核库

**CUTLASS**——NVIDIA 的 CUDA 模板库。高性能 GEMM 的参考实现。模板按架构分别编译；SM100 模板默认目标是 `sm_100a`。见 [`kernels/cutlass`](../kernels/cutlass.md)。

**FlashAttention（FA-2、FA-3）**——Tri Dao 的注意力 kernel。FA-2 可移植；FA-3 仅支持 Hopper（Blackwell 移植开发中）。见 [`kernels/flashattention`](../kernels/flashattention.md)。

**FlashInfer**——面向推理服务的 kernel 库（注意力 + MoE）。有 NVFP4 路径；部分 MoE all-to-all kernel 需要 P2P 原子操作。见 [`kernels/flashinfer`](../kernels/flashinfer.md)。

**DeepGEMM**——DeepSeek 的高吞吐 FP8/FP4 GEMM。发布版本仅支持 SM100。见 [`kernels/deepgemm`](../kernels/deepgemm.md)。

**Marlin**——INT4 权重、FP16 激活的 GEMM。面向较老的架构；SM120 上能用。

**Triton**——用于编写自定义 kernel 的 DSL 编译器。SM120 上能用。

**TransformerEngine**——NVIDIA 的混合精度封装库。SM120 支持仍在完善中。

**NVSHMEM**——基于 NVLink 的单边 GPU 内存原语。要有性能**必须有 NVLink**；有 PCIe 回退路径，但慢到没法用。

**DeepEP**——DeepSeek 的专家并行 a2a kernel。节点内需要 NVLink + NVSHMEM；节点间需要 RDMA。

**vLLM、sglang、TensorRT-LLM**——上层推理引擎。它们把上面这些库组合起来用。见 [`kernels/inference-engines`](../kernels/inference-engines.md)。

## 提到的模型

**DeepSeek-V3 / V4 / V4-Flash**——DeepSeek 的前沿 MoE 家族（671B，V4 有后续演进）。重度依赖 `tcgen05`、DeepGEMM、NVSHMEM、EP。

**Kimi-K2 / K2.6**——月之暗面的 MoE 模型家族。依赖项类似。

**GLM-5.0 / 5.1**——智谱的 MoE 家族（约 478B–744B）。对 `tcgen05` 依赖没那么激进，但参考部署仍在 SM100 上。

**Qwen-3（MoE 变体）**——阿里的开放 MoE 家族。

**REAP**——"REbalanced Activation Pruning"，一种剪枝技术，从 MoE 模型里整个删掉部分专家而质量损失很小。

## 编译流程

**`nvcc`**——NVIDIA CUDA 编译器。前端驱动器，负责主机端 C++ 编译并产出 PTX/cubin。

**`ptxas`**——PTX 汇编器。把 PTX 降到 SASS（cubin）。

**`cuobjdump`**——编译后 CUDA 二进制的检查工具。

**`nvdisasm`**——SASS 反汇编器。

**SASS**——NVIDIA 按架构分别定义的机器码。跨 SM 版本不可移植。

**Cubin**——编译好的 CUDA 二进制。含一个或多个架构的 SASS，可选附带用于 JIT 的 PTX。

**JIT**——即时编译。如果没有匹配的 cubin 段，驱动可以在加载时把 PTX 编译成 SASS。

## 其他

**KV cache**——注意力中的键值缓存。存下历史 token 的 K、V 投影，使每个新 token 的注意力开销是 O(N) 而不是 O(N²)。

**Page-attention / paged KV（分页注意力 / 分页 KV）**——按块管理 KV cache（vLLM 风格）。

**MTP（Multi-Token Prediction，多 token 预测）**——并行预测多个 token 的投机解码方案。

**NSA（Native Sparse Attention，原生稀疏注意力）**——DeepSeek 的稀疏注意力变体。

**REAP-NN**——剪枝后每层保留多少专家的记法（例如 REAP-160 = 256 个专家里保留 160 个）。

**Watchdog（看门狗）**——sglang/vLLM 的后台线程，一次前向传播耗时过长就把服务进程杀掉。

**xid**——NVIDIA 驱动错误码（例如 Xid 79 = GPU 掉出总线）。

**AER（Advanced Error Reporting，高级错误报告）**——PCIe 链路层的错误报告。Gen4 链路负载重时会看到 RxErr 计数增长。

**Bus / function / device IDs（总线 / 功能 / 设备号）**——PCIe 寻址，例如 `01:00.0`。改过 BIOS 设置后可能会变。

## 另见

- [`reference/bibliography`](../reference/bibliography.md)：一手来源
- [`reference/abbreviations`](../reference/abbreviations.md)：一大堆缩写的速查版
