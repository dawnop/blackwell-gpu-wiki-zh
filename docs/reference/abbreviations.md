# 缩写表

全站用到的简称，按字母顺序排列。

| 缩写 | 全称与含义 |
| --- | --- |
| **A2A** | All-to-all，全交换（一种集合通信） |
| **AR** | All-reduce，全规约（一种集合通信） |
| **B100/B200/B300** | NVIDIA Blackwell 数据中心 GPU（SM100） |
| **BW** | Bandwidth，带宽 |
| **CGA** | Cooperative thread-block cluster，协作线程块簇（有时也叫"CTA group"，就是 cluster） |
| **CTA** | Cooperative Thread Array，协作线程数组（PTX 术语里的线程块） |
| **CUTLASS** | CUDA Templates for Linear Algebra Subroutines，CUDA 线性代数模板库 |
| **DSA** | DeepSeek Sparse Attention，DeepSeek 稀疏注意力 |
| **EP** | Expert Parallelism，专家并行 |
| **FA** | FlashAttention |
| **FA2 / FA3** | FlashAttention v2 / v3 |
| **FP4** | 4-bit floating-point，4 位浮点（E2M1） |
| **FP8** | 8-bit floating-point，8 位浮点（E4M3 或 E5M2） |
| **FP16** | 16-bit floating-point，16 位浮点（IEEE 半精度） |
| **FP32** | 32-bit floating-point，32 位浮点（IEEE 单精度） |
| **GB200/GB300** | Grace–Blackwell 超级芯片 |
| **GEMM** | General Matrix-Matrix Multiplication，通用矩阵乘 |
| **GPC** | Graphics Processing Cluster，图形处理簇（GPU 内部的一级划分） |
| **GPU** | Graphics Processing Unit，图形处理器 |
| **GQA** | Grouped-Query Attention，分组查询注意力 |
| **HBM** | High-Bandwidth Memory，高带宽显存（数据中心版 Blackwell 使用） |
| **IB** | InfiniBand |
| **KV** | Key/Value，键/值（transformer 注意力里的） |
| **L1 / L2** | Level-1 / Level-2 cache，一级 / 二级缓存 |
| **MHA** | Multi-Head Attention，多头注意力 |
| **MIG** | Multi-Instance GPU，多实例 GPU |
| **MLA** | Multi-Latent Attention，多头潜在注意力 |
| **MMA** | Matrix-Multiply-Accumulate，矩阵乘累加（Tensor Core 操作） |
| **MoE** | Mixture-of-Experts，混合专家 |
| **NCCL** | NVIDIA Collective Communications Library，NVIDIA 集合通信库 |
| **NSA** | DeepSeek Native Sparse Attention，DeepSeek 原生稀疏注意力 |
| **NVFP4** | NVIDIA FP4 microscaled format，NVIDIA 的 FP4 微缩放格式 |
| **NVL72** | NVLink-72 机柜级互连 |
| **NVSHMEM** | NVIDIA Symmetric Hierarchical Memory，NVIDIA 对称分层内存（面向 GPU 的 PGAS） |
| **OOM** | Out of memory，内存不足 |
| **P2P** | Peer-to-Peer，点对点（GPU 之间直接访问） |
| **PCIe** | PCI Express |
| **PGAS** | Partitioned Global Address Space，分区全局地址空间 |
| **PP** | Pipeline Parallelism，流水线并行 |
| **PTX** | Parallel Thread eXecution，NVIDIA 的 GPU 中间表示 |
| **RDMA** | Remote Direct Memory Access，远程直接内存访问 |
| **REAP** | Router-weighted Expert Activation Pruning，按路由权重的专家激活剪枝 |
| **SASS** | Shader Assembly，NVIDIA 的 GPU 机器码 |
| **SDPA** | Scaled Dot-Product Attention，缩放点积注意力 |
| **SM** | Streaming Multiprocessor，流式多处理器 |
| **SM90 / SM100 / SM120** | 计算能力标签（Hopper / 数据中心版 Blackwell / 工作站版 Blackwell） |
| **SMEM** | Shared Memory，共享内存（每个 SM、每个线程块各自一份） |
| **SXM** | NVIDIA 的高功耗板卡形态（数据中心卡使用） |
| **T** | Tera，太（10¹²）；在 tok/s 语境下指每秒 token 数 |
| **TC** | Tensor Core |
| **TE** | TransformerEngine（NVIDIA 的库） |
| **TFLOPS** | Tera-FLoating-point OPerations per Second，每秒万亿次浮点运算 |
| **TMA** | Tensor Memory Accelerator，张量内存加速器 |
| **TMEM** | Tensor Memory，张量内存（仅 SM100/SM101） |
| **TP** | Tensor Parallelism，张量并行 |
| **VRAM** | Video RAM，显存（即 GPU 设备内存） |
| **W4A16** | 4-bit Weight, 16-bit Activation，4 位权重、16 位激活（一种量化方案） |

带完整文字定义的术语表见 [`overview/glossary`](../overview/glossary.md)。
