# Triton 与 TransformerEngine

两个相邻的库：一个 DSL 编译器（Triton）和一个混合精度的高层封装（TransformerEngine）。两者在工作站 Blackwell 上都用得上，也各有各的坑。

## Triton

一种基于 Python 的 DSL，用来写 CUDA kernel。用户写的是类似对 tile 做 NumPy 运算的高层代码，Triton 编译器把它降级成目标架构的 PTX/SASS。

GitHub：`triton-lang/triton`。最初由 OpenAI 开发，现在由社区维护。支持 SM80 到 SM120。

### 它做什么

你写的东西大概长这样：

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
  pid = tl.program_id(axis=0)
  # ... tile 级别的计算 ...
  a = tl.load(a_ptrs)
  b = tl.load(b_ptrs)
  c = tl.dot(a, b)
  tl.store(c_ptrs, c)
```

Triton 编译器负责：

1. 调度迭代空间，把它映射到 GPU 线程上
2. 分配寄存器和 SMEM
3. 把 `tl.dot` 降级成 `mma.sync`（Hopper 上是 `wgmma.async`；新版本在 SM100 上能降成 `tcgen05.mma`）
4. 处理同步、向量化等底层细节

结果是一个**速度通常能达到手工调优的 CUTLASS 同类 kernel 的 70–90 %**、而**代码量少得多**的 kernel。

### SM 兼容性

Triton 3.0+ 支持 SM80 到 SM120。编译器在编译时根据设备生成适合该架构的 PTX：

- SM80–SM89：`mma.sync`
- SM90：划算的地方用 `wgmma.async`
- SM100：大 tile 用 `tcgen05.mma`，其余用 `mma.sync`（还在推进中，不是所有路径都用上了 `tcgen05`）
- SM120：只有 `mma.sync`，没有 `tcgen05`

SM120 路径支持得很好。从 Triton 3.0 起就有专门针对 SM120 的测试。

SM100 侧的时间线：Triton 3.3 开始对 `tcgen05` 和 TMEM 建模；3.4 去掉了 TMA 描述符 API 的 `_experimental` 前缀（旧的 `tl._experimental_descriptor_*` 在 3.3+ 删除，锁旧版本的 kernel 会编不过），并首次带上 **Gluon**：`triton.experimental.gluon.language.nvidia.blackwell` 直接暴露 `allocate_tensor_memory`、`TensorMemoryLayout`、`tcgen05_mma` / `tcgen05_mma_scaled`，让你在 Python 里按 warp 特化的方式手写 SM100 kernel，而不是靠编译器自动降级。

### 常见故障

- **Triton 版本和下游库不匹配**。FlashInfer、sglang 等都锁定了特定的 Triton 版本。混用会导致导入错误或 kernel 编译失败。
- **kernel 在 SM120 上用了 Hopper 的实验性特性**。有些 Triton kernel 用了 `tl.async_copy`（TMA）或 cluster 功能。在 SM120 上这些会静默降级或报错。维护良好的 Triton kernel 大多会用 `arch.sm > 9` 之类的检查把架构相关特性保护起来。
- **SM120 上的寄存器压力**。SM120 的 SMEM 更小，在 SM100 上能跑的 kernel 到了 SM120 上可能会溢出到本地内存。检测方法：`nvcc --resource-usage`，或者看 `ncu` 的寄存器数量指标。

### 什么时候用 Triton

对工作站 Blackwell 用户：

- **自定义 kernel**：用 Triton 写比写 CUTLASS 等价物快得多，性能也拿得出手
- **带 KV 切分的注意力**：FlashInfer 基于 Triton 的注意力配上 `--triton-attention-num-kv-splits 64`，在长上下文下胜过 FA-2
- **MoE 专家 dispatch**：服务栈里常见用 Triton kernel 实现路由和 combine 逻辑

## TransformerEngine

NVIDIA 的混合精度训练与推理高层库。它给 PyTorch 包上一层支持 FP8/FP4 的模块（Linear、LayerNorm、Attention），自动处理量化、缩放和反缩放。

GitHub：`NVIDIA/TransformerEngine`。协议：Apache-2.0。由 NVIDIA 维护。

### 它做什么

```python
import transformer_engine.pytorch as te

linear = te.Linear(input_dim, output_dim, params_dtype=torch.bfloat16)
with te.fp8_autocast(enabled=True, fp8_recipe=te.recipe.DelayedScaling()):
  y = linear(x)
```

幕后发生的事：

1. 即时把这个 linear 的权重从 BF16 量化到 FP8
2. 在最近若干次调用的窗口内跟踪每个张量的缩放因子
3. 用 CUTLASS 或 cuBLAS-Lt 做真正的 GEMM
4. 反量化回 BF16，交给下游算子

"DelayedScaling" 只是众多 recipe 之一；其他 recipe 面向 FP4、MXFP 或静态缩放的部署。

### SM 兼容性

| 架构 | 支持情况 |
| --- | --- |
| Ampere | 可用（仅 FP16/BF16；FP8 路径会跳过） |
| Hopper | 完整 FP8 支持 |
| 数据中心版 Blackwell（SM 10.0） | 完整 FP4/FP8 支持 |
| 工作站版 Blackwell（SM 12.0） | 部分支持——一些 FP8 路径假定的是 Hopper 级别的语义；FP4 路径还在演进 |

TransformerEngine 对 SM120 的支持是逐步加上的。截至 2026 年初，基础的 Linear 和 LayerNorm 模块在 SM120 上可用；注意力和 MoE 专用模块就没那么可靠了。

### 什么时候用它

对工作站 Blackwell：

- **想给现有 PyTorch 模型加上 FP8 训练或推理**、又不想自己写 kernel，TransformerEngine 是最省事的路
- **纯推理负载别用它**——加载预量化模型这种场景，推理引擎（vLLM、sglang）会直接调 CUTLASS / FlashInfer，TE 那层开销不值得
- 在 SM120 上依赖某个特定 TE 模块之前，**先去看看 GitHub issue**

### 常见故障

- **模块尚未移植到 SM120**。症状：`RuntimeError: Unsupported architecture` 或 `no kernel image`。修法：查 `transformer_engine` 的 issue 跟踪器；有些模块需要手动启用 SM120 后端。
- **FP8 缩放因子过期**。DelayedScaling recipe 维护的是激活缩放因子的滚动历史；推理时调用次数不多，缩放因子可能已经过期。推理请改用 `MXFP8` 或静态 recipe。

## 它们怎么组合在一起

在真实的推理栈里，你可能看到：

```
推理引擎（sglang、vLLM）
  │
  ├─→ FlashInfer（注意力 + MoE）
  │   │
  │   ├─→ Triton kernel（带 KV 切分的注意力、专家路由）
  │   └─→ CUTLASS kernel（NVFP4 GEMM、FP8 GEMM）
  │
  └─→ 自定义 Triton kernel（LayerNorm、RoPE、采样）

训练栈可能还会加上：
  └─→ TransformerEngine（FP8 Linear、FP8 LayerNorm、FP8 注意力）
      │
      └─→ CUTLASS（底层的 GEMM）
```

Triton 的定位是 CUTLASS 的替代：有些 kernel 用 DSL 表达比实例化模板更容易。TransformerEngine 则位于 CUTLASS 之上，是一个对 PyTorch 友好的封装。

## 另见

- [`cutlass`](cutlass.md)——TransformerEngine 封装的东西
- [`flashinfer`](flashinfer.md)——部分注意力路径使用 Triton
- *Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"*（2019）
- GitHub 上的 `triton-lang/triton` 和 `NVIDIA/TransformerEngine`
