# DeepSeek-V3 与 V4

"为 NVL72 而设计、在消费级 Blackwell 上跑不起来"的最典型模型系列。DeepSeek-V3（2024 年 12 月）、V3.1、V4（2026 年 Q1）、V4-Flash（2026 年 Q2）——几款的情况都差不多。

## 模型本身

| | DeepSeek-V3 | DeepSeek-V4 | DeepSeek-V4-Flash |
| --- | --- | --- | --- |
| 总参数 | 671 B | ~700 B | ~250 B |
| 每 token 激活参数 | 37 B | ~40 B | ~25 B |
| 架构 | MoE，共享专家 + 路由专家 | MoE，路由有改进 | MoE，面向低延迟优化 |
| 专家数 | 256 路由 + 1 共享 | 256 路由 + 1 共享 | 64 路由 |
| Top-k 路由 | 8 | 8 | 4 |
| 隐层维度 | 7168 | 7168 | 5120 |
| 层数 | 61 | ~80 | ~40 |
| 原生量化格式 | FP8 | NVFP4 | NVFP4 |
| Attention | MLA（Multi-head Latent Attention） | MLA | MLA |

DeepSeek 的标志是 **MLA**（Multi-head Latent Attention）——一种低秩 attention 变体，KV cache 比标准 MHA 小约 6 倍。再加上它的 MoE：激进的 top-k=8 路由，分到 256 个专家上。

## 参考部署默认了什么

DeepSeek 为 V3/V4 发布的参考部署面向：

- **硬件**：8× H100 / H200（DGX），或 16 张以上的 B100/B200（HGX 或 NVL72）
- **GEMM**：DeepGEMM（他们自己的库），Blackwell 上编译目标是 `sm_100a`
- **Attention**：FlashInfer，带针对 MLA 定制的 kernel
- **MoE all-to-all**：机箱内用 DeepEP intranode（NVSHMEM 走 NVLink），跨节点用 DeepEP internode（RDMA）
- **并行方式**：以 EP 为主（每个专家放在一小组 GPU 上），专家内部用 TP，模型层之间用 PP
- **推理引擎**：vLLM 或 sglang（他们自己的分支）

整套栈都是为**跑在 NVLink 级别 fabric 上的 EP** 设计的。四个主要依赖里有三个（DeepGEMM 的 `tcgen05`、DeepEP 的 NVSHMEM、FlashInfer 的 NVFP4 GEMM）默认数据中心版 Blackwell，其中一个（DeepEP NVSHMEM）还默认有 NVLink。

## 在工作站 Blackwell 上哪里会坏

大致按严重程度排序：

### 1. DeepGEMM 跑不起来

DeepGEMM 的 cubin 只有 `sm_100a` 版本。在 SM120 上加载会报"no kernel image"。

解决：换成 CUTLASS 的 NVFP4 GEMM。推理引擎都有 `disable_deepgemm` 开关。

### 2. DeepEP intranode 要 NVLink，internode 要 RDMA

典型的工作站 Blackwell 机器两样都没有。DeepEP 会直接拒绝初始化。

解决：不用 DeepEP。把推理引擎配成只用 TP。

### 3. 走 PCIe 的 EP 方案根本不可行

就算用 NCCL 回退的 all-to-all，EP=N 在 PCIe Gen4 上也会被带宽卡死。一个 80 层、8 路 EP 的模型，按 PCIe 的速度 decode 大约只有 1–2 tok/s。

解决：用 TP=4（或者你有几张卡就几路）代替 EP。这意味着每张 GPU 都要持有**全部专家权重**，但只算自己那份 TP 切片。内存代价：每张 GPU 上的专家权重必须塞进 `total_VRAM / N`。

### 4. FlashInfer 的 one-shot MoE a2a 需要 P2P 原子操作

如果你非要用 FlashInfer 优化过的 a2a kernel 来跑 EP，它会在完成标志的原子变量上忙等，而这些原子操作根本没开。

解决：同第 3 条。

### 5. NVFP4 缩放因子布局

DeepSeek V4 的 NVFP4 权重用的是一种特定的缩放因子布局。如果加载它的 kernel 期望的是 MX-FP4，或者是另一种布局的 NVFP4，输出会是一堆无声的垃圾。

解决：确认推理引擎的量化配置和权重文件一致（检查模型目录里的 `quantization_config.json` 以及引擎的加载器）。

### 6. MLA 的 KV cache 特殊之处

MLA 把 key 和 value 存成低秩的潜向量，用的时候才即时展开。参考实现里这个展开 kernel 有 `tcgen05` 路径。

解决：大多数推理引擎都有基于 `mma.sync` 的 MLA 展开 kernel（就是 DeepSeek-V2 论文附带的 C++ 参考实现）。它在 SM120 上能跑，只是慢一些。

## 在工作站 Blackwell 上什么能用

一套能跑的配置：

```yaml
# 示意写法——请换成你的引擎的参数语法
weights: NVFP4 (DeepSeek's published artifacts)
kv_cache: FP8 E4M3
attention: Triton-based with kv_splits=64
parallelism:
  tensor_parallel: 4 (= number of GPUs)
  pipeline_parallel: 1
  expert_parallel: 1     # 关掉 EP
gemm_backend: cutlass    # 不用 deepgemm
moe_runner: cutlass      # 不用 flashinfer one-shot
```

V4-Flash（250 B 参数，64 个专家）用这套配置能轻松塞进 4× 96 GB 的工作站 Blackwell，还能留出约 32k 上下文的 KV cache 空间。完整的 V4（700 B）则需要先做剪枝（REAP 那种），把参数量降到约 478 B，再做 NVFP4 量化才放得下。

## 性能预期

在 4 张工作站 Blackwell 上，用上面的变通配置：

| 模型 | 预期 decode tok/s |
| --- | ---: |
| V4-Flash | 50–80（模型小，放得宽裕） |
| V4（REAP 剪枝后） | 20–50（随上下文长度变化） |
| V3（FP8 权重，不剪枝） | 10–30（FP8 权重比 V4 的 NVFP4 大） |

对比同等卡数的 B100 部署 V4 大约 200 tok/s，这里大约**慢 5–10 倍**。差距一部分来自硬件（显存带宽、SM 数量），一部分来自没有最优 kernel（基于 `tcgen05` 的 GEMM 对比基于 `mma.sync` 的 GEMM）。

## DeepSeek 特殊在哪

几点让 DeepSeek 成为"消费级 Blackwell 跑不了"的典型代表：

1. **他们自己写了 GEMM 库**（DeepGEMM），而且只编 `sm_100a`。大多数实验室用 CUTLASS，两个目标都有。
2. **MLA 不常见。**外面的 attention kernel 大多是标准 MHA / GQA；MLA 需要专门的 kernel 路径，不是每个库都提供。
3. **模型太大。**即使用 NVFP4 也得一个字节一个字节地抠。
4. **参考文档默认数据中心环境。**他们的部署指南有 DGX H100 的逐步说明，工作站卡则一个字没有。

如果你能在工作站 Blackwell 上跑起 DeepSeek V4，那大概什么模型都能跑了。

## 另见

- [`kimi-k2`](kimi-k2.md) —— 情况类似，另一个团队
- [`glm-5`](glm-5.md) —— 对 `tcgen05` 依赖更轻，更容易移植
- [`compatibility/`](../compatibility/index.md) —— 各种变通方法的通用套路
- DeepSeek-V3 和 V4 技术报告
- GitHub 上的 `deepseek-ai/DeepGEMM`、`deepseek-ai/DeepEP`
