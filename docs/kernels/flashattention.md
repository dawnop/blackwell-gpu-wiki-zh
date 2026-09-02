# FlashAttention

让长上下文 transformer 变得可行的那个注意力 kernel。三代：FA-1（2022，基础版）、FA-2（2023，针对 Ampere/Hopper 调优）、FA-3（2024，Hopper 异步版）。截至 2026 年初，FA-3 还没有 Blackwell 移植；FA-2 有。

## 是什么

一个计算缩放点积注意力 `softmax(QKᵀ/√d)V` 的 CUDA kernel，全程不把完整的 N×N 注意力矩阵落到内存里。核心思路：

1. **分块**：把 Q、K、V 切成小到能放进 SMEM 的 tile 来处理
2. **在线 softmax**：一边处理 tile 一边增量地算 softmax，永远不生成完整的注意力矩阵
3. **反向时重算**：不存注意力矩阵，反向传播时重新算一遍

结果是**内存 O(N)** 而不是 O(N²)，实际耗时比朴素注意力**快 2–4 倍**。对长上下文推理来说，这就是能跑和不能跑的分界线。

GitHub：`Dao-AILab/flash-attention`。作者：Tri Dao 及合作者。MIT 协议。

## 依赖什么

- CUDA toolkit
- C++17
- FA-2：Ampere 到 Blackwell 都能跑
- FA-3：需要 Hopper（按 Hopper 扩展方式设计的数据中心版 Blackwell 也算）

这个 kernel 基本是自包含的——运行时不依赖 CUTLASS 或其他库，不过 FA-3 内部用了 CUTLASS 风格的抽象。

## SM100 的情况

**FA-2** 通过通用 kernel 路径跑在 SM100 上，吞吐说得过去但不是最优（走的是 `wgmma.async` 和 `mma.sync` 路径，没有用 `tcgen05`）。

**FA-3** 支持 Hopper（它的主要目标），也能在 SM100 上跑，因为 SM100 大体上是 Hopper 的扩展（cluster、TMA、FP8）。但 FA-3 还没有利用 `tcgen05`——FA-3 在 SM100 上相对 FA-2 的提速和它在 Hopper 上差不多（约 30 %），远不到基于 `tcgen05` 重新设计后能带来的那种大幅提升。

一个使用 `tcgen05.mma` 的"FA-Blackwell"移植正在开发中。NVIDIA 的 TransformerEngine 里有一个候选实现；社区也在积极推进。

## SM120 的情况

**FA-2** 能在 SM120 上跑，但吞吐有所下降（没有异步 Tensor Core 重叠；只有 `mma.sync`；99 KiB 的 SMEM 上限导致 tile 更小）。

**FA-3** 不以 SM120 为目标（它的 kernel 结构假定了 Hopper 级别的 SM 特性，工作站 Blackwell 大部分有但不是全都有）。移植是可行的，但目前没人在做。

对大多数工作站 Blackwell 用户来说，实际的答案是**用 FA-2**（或者 FlashInfer 基于 Triton 的注意力 kernel 之一——见 [`flashinfer`](flashinfer.md)）。

## 常见故障

**故障 1：SM120 上 `flash_attn` 导入报错 / 构建失败**

`flash-attn` Python 包是针对特定架构目标编译的。默认的 wheel 可能不含 `sm_120` 的 cubin。

修复：显式设置 `TORCH_CUDA_ARCH_LIST="12.0"` 从源码构建，或者用一个包含 `sm_120` 的预编译 wheel（项目从 2025 年第四季度起提供这样的 wheel）。

**故障 2：长上下文 decode 比预期慢**

FA-2 在 SM120 上用的 tile 比 SM100 上小，所以每一步的吞吐更低。对长上下文 decode（batch=1，瓶颈在读 KV 的内存带宽上）来说，这一点很伤。

变通办法：用基于 Triton 的注意力，加上 `--triton-attention-num-kv-splits 64`（或等价参数），它通过把 KV 循环激进地并行化，在 SM120 上长上下文场景下可以超过 FA-2。有些推理引擎在 SM120 上默认就是这么做的。

**故障 3：`cu_seqlens` 索引不一致**

FA-2 一个由来已久的坑：变长注意力（`flash_attn_varlen_func`）接收 `cu_seqlens_q` 和 `cu_seqlens_k` 两个数组。如果它们和上层服务栈（vLLM、sglang）的 `qo_indptr` / `kv_indptr` 对不上，注意力就会从一个序列的末尾越界读到下一个序列里。

症状：注意力输出大体正确，但混入了并发请求之间的泄漏。

修复：确保 `cu_seqlens` 和请求批次的布局严格一致。SGLang 带了一个补丁（`qo_indptr` 索引修复），这个问题是在早期的 SM120 部署中暴露出来的。

## 检测方法

检查 `flash-attn` 是否已安装、它的 kernel 面向哪些架构：

```bash
python -c "import flash_attn; print(flash_attn.__file__)"
# 然后对那里引用的 .so 执行：
cuobjdump --list-elf $(python -c "import flash_attn; ...") | grep arch
```

检查 Python API 会分发到哪些 kernel：

```python
import flash_attn
print(flash_attn.__version__)            # 2.7.x 或 3.x
# 3.x 表示 FA-3 路径已启用
```

## 接下来会怎样

原生 Blackwell 的注意力 kernel（QKᵀ 和 softmax-V 两个 GEMM 都用 `tcgen05.mma`）是 FA 家族最受期待的新成员。落地时大概会叫 FA-3.x 或 FA-4。在 B100 上，长序列相对 FA-2 的预期提速很大（2–3 倍）。

工作站 Blackwell 会间接受益：一个面向工作站 Blackwell 的 FA-3 移植（不用 `tcgen05`）凭借 FA-3 引入的更好的流水线结构，在 SM120 上相对 FA-2 仍会是有意义的提升。这样的移植会不会有人去做，取决于社区的需求。

## 另见

- [`flashinfer`](flashinfer.md) — 服务场景的其他注意力 kernel，包括基于 Triton 的长上下文路径
- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) — `mma.sync` vs `wgmma.async` vs `tcgen05.mma`
- *Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*（2022）
- *Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"*（2023）
- *Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"*（2024）
- GitHub 上的 `Dao-AILab/flash-attention`
