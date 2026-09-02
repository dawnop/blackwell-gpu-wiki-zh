# FlashInfer

面向服务场景的 kernel 库——注意力和 MoE 都做，重点是生产推理中遇到的各种不规则形状（变长序列、分页的 KV cache、MoE 路由）。

GitHub：`flashinfer-ai/flashinfer`。协议：Apache-2.0。由一个学术界与工业界联合的团队维护。

## 是什么

一个屋檐下的两个独立 kernel 家族：

- **注意力 kernel**：分页注意力、前缀注意力、decode 注意力、KV 切分注意力。有基于 Triton 和基于 CUTLASS 两种变体。
- **MoE kernel**：top-k 路由、token dispatch、专家 FFN 执行、输出 combine。单 GPU 和多 GPU（带 all-to-all）都有。

sglang、vLLM（部分路径）和其他几个推理引擎都用它作为注意力后端。

## 依赖什么

- CUDA toolkit
- PyTorch（Python 绑定需要）
- Triton（基于 Triton 的 kernel 需要）
- CUTLASS（基于 CUTLASS 的 kernel 需要，包括 NVFP4 路径）

有自己的 JIT 缓存（一般在 `~/.cache/flashinfer/<version>/<arch>/cached_ops/`），首次使用时编译 kernel 变体，之后直接复用。缓存目录名里编码了架构，例如 SM120 对应 `120a`。

## SM100 的情况

完整支持。FlashInfer 的 NVFP4 路径用的是面向 `sm_100a`、基于 `tcgen05.mma` 的 CUTLASS 模板。MoE 的 one-shot all-to-all 在有 MNNVL fabric 原语的地方用它，否则回退到 NCCL。

## SM120 的情况

好坏参半：

- **注意力**：可用。基于 Triton 的 kernel 跨架构可移植；基于 CUTLASS 的变体能为 SM120 编译，但 tile 要缩小。
- **NVFP4 GEMM**：通过 FlashInfer 的 CUTLASS-NVFP4 路径在 SM120 上可用，但要注意 SMEM 断崖（用更小的 tile 形状）。
- **FP4 GEMM 的 CuDNN 后端**：在 SM120 上可用（CuDNN 本身两种架构都支持）。
- **MoE one-shot all-to-all**：没有 P2P 原子操作的话**在消费级 Blackwell 上跑不起来**。见下文。

## MoE one-shot all-to-all 问题

FlashInfer 的"one-shot"MoE all-to-all kernel 是一个高性能变体，流程是：

1. 每个 rank 有一块要 dispatch 给其他 rank 的激活值缓冲区
2. 每个 rank 用 P2P 写直接写进对端 rank 的目标缓冲区
3. 每个 rank 忙等轮询存放在对端 rank 内存里的**完成标志**
4. 一旦标志表明所有对端都写完了，kernel 就进入 combine 阶段

第 3 步在消费级 Blackwell 上出了问题。完成标志是用**原子写**更新的——而消费级 GPU 上的 PCIe 原子操作默认被软件关掉了。于是轮询的 rank 永远看不到标志变成"已完成"。

这个 kernel 内部有 60 秒超时。60 秒之后：

```
flashinfer error: Rank 0 timed out waiting for completion flag from rank 1
flashinfer error: Rank 1 timed out waiting for completion flag from rank 2
... (similar from all ranks)
cudaFuncSetAttribute ... unspecified launch failure at cutlass_fused_moe_kernels.cuh:417
```

然后服务就崩了。

**修复路径 1**：打开 P2P 原子操作。两样都得做：

- BIOS：ACS Enable → **Disabled**
- 驱动：通过 `/etc/modprobe.d/` 设置 `NVreg_RegistryDwords="RMDisableFeatureDisablement=1"`

在工作站硬件上这两项通常都被锁死了；尤其是关闭 ACS，取决于主板 BIOS 买不买账。

**修复路径 2**：换一个 MoE all-to-all kernel。基于 NCCL 的 all-to-all（FlashInfer 有回退路径）不依赖原子操作，但更慢，而且有时会在预热阶段死锁。

**修复路径 3**：干脆不用 all-to-all。用张量并行方案代替专家并行。见 [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md)。

## 常见故障

**故障 1：NVFP4 + SMEM 断崖**

FlashInfer 里基于 CUTLASS 的 NVFP4 GEMM 会撞上和原生 CUTLASS 一样的 SMEM 断崖。缓解办法：确保 kernel 的 tile 形状是按 SM120 的 SMEM 定的。近期版本里，FlashInfer 的 `120a` JIT 缓存默认应该会选对 tile 形状。

**故障 2：JIT 缓存损坏**

JIT 缓存偶尔会缓存一次坏的编译结果（例如来自上一个库版本）。症状：之前好好的 kernel 开始输出错误结果。

修复：`rm -rf ~/.cache/flashinfer/`，让它重新构建。

**故障 3：Triton 版本不匹配**

FlashInfer 基于 Triton 的 kernel 需要特定范围的 Triton 版本（截至 2025 年末是 3.0–3.2）。更老或更新的 Triton 会导致导入报错或 kernel 编译失败。

修复：在环境里锁定 Triton 版本；或者，如果你清楚自己在干什么，设置 `FLASHINFER_DISABLE_VERSION_CHECK=1`。

**故障 4：page size 不匹配**

FlashInfer 的 KV cache 注意力接收一个 `page_size`（一般是 16、64 或 128）。有些 kernel 路径只支持特定的 page size：

- Decode-NSA 路径：要求 `page_size=64`
- Prefill-MTP 路径：要求 `page_size=64`
- 标准 decode：{16, 64, 128} 都行

如果服务栈在 `page_size=128` 下尝试用 NSA 或 MTP，FlashInfer 会在启动时断言失败。修复在上游：把推理引擎配置成 `page_size=64`。

## 检测方法

```bash
python -c "import flashinfer; print(flashinfer.__version__, flashinfer.__file__)"

# 找到缓存的 kernel：
ls -la ~/.cache/flashinfer/<version>/<arch>/cached_ops/
# 有 120a 说明存在为 SM120 缓存的 kernel
```

存在 `120a/cached_ops/` 就说明 FlashInfer 在这台设备上用过，并且编译了 SM120 kernel。查看具体的 kernel：

```bash
ls ~/.cache/flashinfer/0.6.7/120a/cached_ops/
# fused_moe_120, decode_attention_120, prefill_attention_120, ...
```

## 阅读 FlashInfer 源码

```
flashinfer/
├── csrc/                       # C++ + CUDA kernel 实现
│   ├── attention/              # 注意力 kernel
│   ├── moe/                    # MoE all-to-all + 专家 kernel
│   ├── gemm/                   # 基于 CUTLASS 的 NVFP4 GEMM
│   └── jit/                    # JIT 运行时
├── flashinfer/                 # Python 绑定
└── ...
```

MoE one-shot a2a kernel 在 `csrc/moe/all_to_all.cuh`。原子忙等轮询在接近末尾的等待循环里。

## 另见

- [`cutlass`](cutlass.md) — FlashInfer 的 NVFP4 路径底层用的东西
- [`flashattention`](flashattention.md) — 专讲注意力
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md) — 原子操作问题的详细说明
- [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) — EP 和 TP 之间的选择
- GitHub 上的 `flashinfer-ai/flashinfer`
- *Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving"*（2024）
