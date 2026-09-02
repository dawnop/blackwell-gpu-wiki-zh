# Kimi-K2 系列

Moonshot AI 的 MoE 系列。2025–2026 年陆续发布（K2.0 → K2.6+）。和 DeepSeek-V3/V4 一样是 MoE 加 NVFP4 量化，情况类似，但有几处差别会影响它在工作站 Blackwell 上的表现。

## 模型本身

| | K2.0 | K2.6 |
| --- | --- | --- |
| 总参数 | ~600 B | ~700 B |
| 每 token 激活参数 | ~32 B | ~40 B |
| 专家数 | 384 | 384（路由有改进） |
| Top-k | 8 | 6 |
| 隐层维度 | 6144 | 7168 |
| Attention | GQA（Grouped-Query Attention） | GQA |
| 原生量化格式 | FP8 → NVFP4（后续版本） | NVFP4 |

Kimi 用的是标准 GQA，而不是 DeepSeek 的 MLA。每 token 的 KV cache 比 DeepSeek MLA 大，但 attention kernel 到处都有（FlashAttention-2 原生支持 GQA）。

## 参考部署默认了什么

Moonshot 给 K2 的部署指南面向：

- **硬件**：H200 / B100 / B200，最好有 NVLink
- **GEMM**：他们自己的 CUTLASS 分支，带定制的 NVFP4 模板，编译目标 `sm_100a`
- **Attention**：FlashAttention-2 或 FlashInfer（GQA 路径）
- **MoE**：vLLM，all-to-all 用 FlashInfer-MoE
- **并行方式**：专家用 EP，专家内部用 TP，模型层之间用 PP

依赖面**比 DeepSeek 轻**：Kimi 没有自带 GEMM 库，用的是 CUTLASS；attention 用的是 FlashAttention-2（可移植），而不是定制 MLA kernel。MoE 的 all-to-all 是唯一一处重要的、只有 SM100 才满足的假设。

## 在工作站 Blackwell 上哪里会坏

### 1. CUTLASS NVFP4 路径撞上 SMEM 断崖

Moonshot 的 CUTLASS 模板继承了上游 CUTLASS 对 SMEM 预算的假设。在 SM120 上，自动 carveout 申请的量超过 99 KiB，会把 SMEM bank 写坏。

解决：要么改 CUTLASS 模板，显式把 `StageCount` 设小；要么用上游面向 SM120 的模板（tile 形状和 Moonshot 的默认值略有不同）。

### 2. EP + FlashInfer a2a 在 PCIe 原子操作上出问题

和 DeepSeek 一样。用 NCCL 回退，或者改成只用 TP。

### 3. 384 个专家让纯 TP 的显存很紧

384 个专家，每个 NVFP4 下约 1.5 GB，专家权重总共约 570 GB。在 4× 96 GB 的机器上（共 384 GB），TP=4 且专家复制（每张 GPU 持有全部 384 个专家的 TP 切片）需要 570/4 ≈ 143 GB 每卡——96 GB **放不下**。

这就是纯 TP 行不通、必须上混合方案的情况：TP=4 + PP=2（把层拆到两组 GPU 上），或者忍着带宽代价用 EP + NCCL。

对 K2.6，通常需要下面之一：

- 剪枝到约 256 个专家（REAP 那种）→ 纯 TP 放得下
- TP × PP 混合方案（每 token decode 变慢，但显存压力小）
- 接受 EP + NCCL（慢，但能跑）

### 4. 路由 kernel 的小问题

K2 用了一个定制的 top-k 路由 kernel，路由 softmax 默认按 `tcgen05` 那种异步 Tensor Core 方式执行。在 SM120 上会回退到较慢的路径。不影响正确性，只是性能损失。

## 能跑的配置

```yaml
weights: NVFP4 (Moonshot's K2.6 release, 256 experts after REAP pruning)
kv_cache: FP8 E4M3
attention: FlashAttention-2 (GQA path) or Triton fallback
parallelism:
  tensor_parallel: 4
  pipeline_parallel: 1 (or 2 if memory tight)
  expert_parallel: 1
gemm_backend: cutlass with explicit StageCount=2 or 3
```

注意：因为 FlashAttention-2 对 GQA 开箱即用，Kimi-K2 在工作站 Blackwell 上某些方面反而比 DeepSeek-V4 **更好跑**，尽管两者规模相当。

## 性能预期

4 张工作站 Blackwell 上：

| 变体 | Decode tok/s |
| --- | ---: |
| K2.6，REAP 剪枝到 256 个专家 | 30–50 |
| K2.6，完整 384 个专家，TP × PP=2 | 15–30 |
| K2.6，EP + NCCL | 5–10 |

数据中心 B100 部署能到 100–200 tok/s。差距和 DeepSeek-V4 差不多：约 5 倍。

## Kimi 的特别之处

- **没有自研 GEMM 库**：依赖 CUTLASS，而 CUTLASS 支持 SM120。移植更容易。
- **标准 GQA attention**：每个 kernel 库都能跑。
- **专家数量多**：显存压力比 DeepSeek 大（256 → 384）。
- **路由 kernel 依赖 `tcgen05` 风格**：这是 kernel 层面的依赖，不是模型架构层面的。

如果你能在工作站 Blackwell 上跑 DeepSeek-V4，那 Kimi-K2 肯定也能跑——唯一的例外可能是 384 个专家带来的显存压力。

## 另见

- [`deepseek-v3-v4`](deepseek-v3-v4.md) —— 情况相当，用 MLA attention
- [`generic-moe-on-consumer-blackwell`](generic-moe-on-consumer-blackwell.md) —— 总结
- [`compatibility/ep-to-tp-rewriting`](../compatibility/ep-to-tp-rewriting.md) —— 并行方案改写的套路
- Moonshot AI 的 K2 发布博客和 HuggingFace 模型卡
