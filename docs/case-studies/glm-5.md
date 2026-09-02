# GLM-5 系列

智谱 AI / 清华的 MoE 系列。GLM-4.5、GLM-4.7、GLM-5.0、GLM-5.1（2025–2026）。这是公认**在工作站 Blackwell 上最好跑的大型 MoE**，原因在于它的依赖栈结构。

## 模型本身

| | GLM-5.0 | GLM-5.1（完整版） | GLM-5.1（REAP-160） |
| --- | --- | --- | --- |
| 总参数 | ~620 B | 744 B | 478 B |
| 每 token 激活参数 | ~36 B | 42 B | ~36 B |
| 专家数 | 160 路由 | 256 路由 | 160 路由（剪枝后） |
| Top-k | 6 | 8 | 8 |
| 隐层维度 | 6144 | 7168 | 7168 |
| Attention | MHA + 可选 DSA | MHA + 可选 DSA | MHA + 可选 NSA |
| 原生量化格式 | FP8 | NVFP4 | NVFP4 |

GLM-5.1 引入了两种 attention 变体：标准 MHA，以及面向长上下文的 DSA（Differential Sparse Attention）。DSA 在 SM120 上有一个已知的 kernel 问题（见下文）；MHA 没问题。

REAP-160 是智谱官方的剪枝配方，把 GLM-5.1 从 256 个专家减到 160 个，质量损失很小（困惑度约变差 0.3）。得到的 478B 参数模型**尺寸就是按 4× 96 GB 工作站 Blackwell 来定的**。

## 参考部署默认了什么

智谱给 GLM-5 的部署指南面向：

- **硬件**：H200 / B200，最好有 NVLink
- **GEMM**：原版 CUTLASS 的 NVFP4 模板
- **Attention**：FlashAttention-2（MHA）或定制 DSA kernel
- **MoE**：sglang 或 vLLM，由引擎在 FlashInfer-MoE 和 CUTLASS-MoE 后端之间选择
- **并行方式**：专家用 EP=N，专家内部用 TP

依赖栈是**本节三个系列里最轻的**：原版 CUTLASS、原版 FlashAttention，没有自研 GEMM 库。唯一依赖 `tcgen05` 的代码路径是可选的 FlashInfer-MoE one-shot a2a，需要手动开启。

## 在工作站 Blackwell 上哪里会坏

### 1. EP 方案（某些配置的默认值）

如果推理引擎默认走 EP，就会撞上 all-to-all 的带宽墙。改成 TP。

### 2. SM120 上的 DSA 路径

GLM-5.1 的 DSA kernel 有一条路径在某些推理引擎版本里无法在 SM120 上编译。这是已知问题；vLLM 0.7.x 和 sglang 0.5.10+ 已经修了。

长上下文推理可以二选一：

- 用打过补丁的引擎开启 DSA
- 回退到 MHA（长上下文下更慢，但结果正确）

### 3. CUTLASS SMEM 断崖

和别处一样——CUTLASS 的 NVFP4 模板可能把 SMEM 撑爆。用 SM120 模板，tile 设小一点。

### 4. 某些 sglang 版本上的 fp8_e4m3 KV

一些中间版本的 sglang 对 FP8 KV 的反量化会走基于 `tcgen05.cp` 的路径。请固定到 **sglang 0.5.10.post1 或更高版本**，它的反量化路径基于 `mma.sync`。

## 在工作站 Blackwell 上什么能用

GLM-5.1 REAP-160 NVFP4 的一套能跑的配置：

```yaml
weights: NVFP4 (REAP-160 prune of GLM-5.1, fits 4× 96GB)
kv_cache: FP8 E4M3
attention: Triton-based with kv_splits=64
parallelism:
  tensor_parallel: 4
  pipeline_parallel: 1
  expert_parallel: 1
gemm_backend: cutlass (SM120 templates)
attention_backend: triton (auto-selected on SM120)
```

GLM-5.1 REAP-160 + NVFP4 量化能装进 4× 96 GB 工作站 Blackwell，还留有充足的 KV cache 余量——用 FP8 KV 足够跑 **200k token 上下文**。对这一类模型来说，工作站 Blackwell 是真正可用的部署目标。

## 性能预期

4 张工作站 Blackwell 上：

| 变体 | 上下文 | Decode tok/s |
| --- | ---: | ---: |
| GLM-5.1 REAP-160 | 256 | 46 |
| GLM-5.1 REAP-160 | 4 K | 42 |
| GLM-5.1 REAP-160 | 16 K | 38 |
| GLM-5.1 REAP-160 | 150 K | 22 |

这些数据来自 TP=4、NVFP4 权重、FP8 KV、kv_splits=64 的配置。如果不设 kv_splits=64，长上下文那一行会掉到约 5 tok/s——在这类硬件上，这一个开关就是最大的单项性能杠杆。

对比一下：同一模型、同一配置在 4× B100（数据中心版 Blackwell）上大约 120–150 tok/s。约 5 倍的差距在所有案例分析里都一致。

## 为什么 GLM-5 是"容易"的那个

- **定制 kernel 面更小**：原版 CUTLASS、原版 FlashAttention，没有 DeepGEMM 之类的东西。
- **剪枝配方按消费级硬件定尺寸**：REAP-160 明确瞄准 4× 96 GB。
- **MoE 并行方案灵活**：模型架构在纯 TP 下工作良好，不强制 EP
- **社区活跃**：工作站 Blackwell 部署有文档、有人测过

如果你刚开始在工作站 Blackwell 上跑 MoE，想找一个"直接能用"的模型，推荐从 GLM-5.1 REAP-160 NVFP4 起步。

## MTP 和 NSA 带来什么

- **MTP（Multi-Token Prediction，多 token 预测）**：投机解码，提高吞吐。多加一个预测头；要求 `page_size=64` 和 BF16 KV。可选，通过推理引擎参数开启。
- **NSA（Native Sparse Attention，原生稀疏注意力）**：在 attention 计算里引入稀疏性，面向长上下文。某些版本在 SM120 上有 kernel 问题；最稳妥的做法是关着。

初次部署时两个都别开。如果基线配置的吞吐不够用，再开 MTP。

## 另见

- [`deepseek-v3-v4`](deepseek-v3-v4.md) —— 依赖栈更重
- [`kimi-k2`](kimi-k2.md) —— 规模相当，显存压力更大
- [`compatibility/`](../compatibility/index.md) —— 通用套路
- 智谱 AI 的 GLM-5 发布说明和 HuggingFace 模型卡
- REAP 剪枝论文
