# tcgen05 与 Tensor Memory

仅数据中心版 Blackwell 才有的新 Tensor Core 指令族，以及它所依赖的那种片上存储。

## `tcgen05` 是什么

`tcgen05` 是 PTX ISA 8.4 引入（8.5 又做了完善）的一族 PTX 指令，只面向计算能力 10.0（即 SM100）。`5` 指第 5 代 Tensor Core，`gen05` 表示这是第 5 代专属的。它的设计目标：

1. **把 Tensor Core 的执行和 warp 的执行解耦。** warp 发射一条 MMA 后继续往下走；Tensor Core 在旁边并行跑到结束。
2. **支持比 `wgmma.async` 更大的 MMA tile。** 单 CTA 最大 128×128，CTA pair 最大 256×128。
3. **减轻寄存器堆的带宽压力。** 累加器放在 TMEM 里，而不是寄存器里。

这几条加起来，在同一个 SM 上，相比基于 `wgmma.async` 的 kernel，FP4/FP6/FP8 峰值吞吐大约能到 **2–3 倍**。

## 指令一览

| 指令 | 作用 |
| --- | --- |
| `tcgen05.alloc.cta_group::N %dst, N` | 分配 N 字节 TMEM，基地址返回到 `%dst` |
| `tcgen05.dealloc %addr, N` | 释放 `%addr` 处的 N 字节 |
| `tcgen05.relinquish_alloc_permit` | 告诉运行时这个 CTA 不会再分配 TMEM |
| `tcgen05.cp.shared::cta::tmem.b64 [%tmem], [%smem]` | 从 SMEM 拷到 TMEM |
| `tcgen05.cp.tmem.shared::cta.b64 [%smem], [%tmem]` | 从 TMEM 拷到 SMEM |
| `tcgen05.shift [%tmem], shift_amount` | 在一段 TMEM 分配内做逻辑移位（用于布局变换） |
| `tcgen05.mma.cta_group::1.kind::<dtype>` | 单 CTA MMA |
| `tcgen05.mma.cta_group::2.kind::<dtype>` | CTA pair MMA |
| `tcgen05.commit.cta_group::N %sema` | 提交一个屏障，用来等待所有未完成的 MMA |
| `tcgen05.wait.cta_group::N %sema` | 等待之前提交的屏障 |

`<dtype>` 列举支持的 MMA 类型：`f4`、`mxf4`、`nvf4`、`f6`、`f8f6f4`、`f8`、`f16`、`bf16`、`tf32` 等。

## 一段完整的 `tcgen05` MMA PTX

一个简化的数据中心版 Blackwell GEMM tile：

```ptx
.reg .b64 %tmem_base;
.reg .b32 %sema;

// 1. 为累加器分配 16 KB TMEM
tcgen05.alloc.cta_group::1 %tmem_base, 16384;

// 2. （操作数 A、B 已经由 TMA 暂存到 SMEM）

// 3. 发射 MMA：D = A * B（FP4 输入，FP32 累加器在 TMEM 中）
tcgen05.mma.cta_group::1.kind::nvf4
    [%tmem_base],          // 累加器
    [%smem_a],             // 操作数 A（在 SMEM 中）
    [%smem_b],             // 操作数 B（在 SMEM 中）
    %scale_a, %scale_b;    // NVFP4 缩放因子寄存器

// 4. 提交屏障
tcgen05.commit.cta_group::1 %sema;

// 5. MMA 运行期间继续做别的事
// ... 更多代码 ...

// 6. 等待 MMA 完成
tcgen05.wait.cta_group::1 %sema;

// 7. 把结果从 TMEM 拷到 SMEM，供后续使用
tcgen05.cp.tmem.shared::cta.b64 [%smem_out], [%tmem_base];

// 8. 释放 TMEM
tcgen05.dealloc %tmem_base, 16384;
tcgen05.relinquish_alloc_permit;
```

和 `wgmma.async` 对比：

- `wgmma` 的累加器在**寄存器**里；`tcgen05` 的累加器在 **TMEM** 里
- `wgmma` 由一个 **warp group**（4 个 warp）发射；`tcgen05` 由**单个 warp**（或一个 CTA pair）发射
- `wgmma` 的 tile 最大到 m64n256k16；`tcgen05` 能到 m128n128k64（单 CTA）或 m256n128k64（CTA pair）
- 两者都是异步的，都有 commit/wait 屏障

## Tensor Memory（TMEM）

一种新的片上存储。特性：

- **容量**：每个 SM 256 KB
- **分配粒度**：128 字节
- **分配器**：`tcgen05.alloc` 返回 TMEM 基地址，`tcgen05.dealloc` 释放
- **寻址**：TMEM 地址与 SMEM 和全局地址**相互独立**——它是每个 SM 的 TMEM 区域内的 32 位逻辑地址
- **带宽**：足以让 `tcgen05.mma` 以峰值速率运行
- **可见性**：TMEM 是每个 SM 私有的；发射指令的 CTA（或 CTA pair）可以寻址它，其他 CTA 不行

TMEM 的存在只为一个原因：在 FP4/FP6 MMA 的吞吐量级上，对 `wgmma.async` 风格的 kernel 来说，**寄存器堆的带宽成了瓶颈**。Tensor Core 消耗操作数的速度，超过了一个 warp 的寄存器读取所能供给的速度。把累加器挪出寄存器（把操作数暂存也挪出寄存器，先进 SMEM 再进 TMEM）之后，warp 的寄存器堆就只需要服务**发射和收尾**两个阶段，而不用伺候运行中的 MMA。

一个好用的心智模型：TMEM 之于 Tensor Core，就像 L1 缓存之于 ALU。

### TMEM 布局

`tcgen05` 支持几种 TMEM 布局：

- **默认**：每 32 个元素一带，带内按行主序
- **步长**：可配置步长，用于转置访问
- **复制**：同一个逻辑操作数复制到多个 TMEM 区域，供 CTA pair MMA 使用
- **压缩**：感知 NVFP4 的布局，把数值和缩放因子交错存放

`tcgen05.shift` 指令负责在这些布局之间原地转换。

## CTA pair / `cta_group::2` 模式

`tcgen05.mma` 最大的 tile（m256n128k64）太大，一个 CTA 的 TMEM 预算装不下。`cta_group::2` 模式让两个 CTA 协作：

- 两个 CTA 作为同一个 **cluster** 的成员启动（`.cluster_dim 2,1,1`）
- 它们通过 cluster 的 SMEM 链路共享 TMEM 分配
- 只发射一条 `tcgen05.mma.cta_group::2`（两个 CTA 步调一致地发射），产出的 tile 是单 CTA 模式的两倍大

这是 SM100 支持大于 1 的线程块簇的原因之一：`tcgen05` 的 CTA pair 模式需要它。

**工作站版 Blackwell 不支持大于 1 的 cluster，因此完全无法使用 `tcgen05` 的 CTA pair MMA。** 为 SM120 编译的 kernel 只能用单 CTA 的 tile 形状——或者更常见的情况是，根本不用 `tcgen05`。

## 为什么工作站版 Blackwell 没有 `tcgen05`

NVIDIA 大概的考虑（从架构反推）：

- TMEM 很占芯片面积（每 SM 256 KB 是实打实的硅片）
- cluster 执行需要额外的 SM 到 SM 链路（又是硅片）
- 消费级负载（游戏、内容创作、轻量 ML）从 m128n128k64 的 GEMM 里得不到多少好处
- 把数据中心和消费级区分开是有意为之的产品策略

结果就是：工作站版 Blackwell 有着**同样的 Tensor Core 硬件**（第 5 代，原生 FP4/FP6/FP8），但只能通过 `mma.sync` 和 `wgmma.async` 访问，而这两者都受寄存器限制。所以它每 SM 的 FP4 峰值吞吐和 Hopper 每 SM 的 FP8 吞吐差不多——有用，但没有 SM100 那种 2–3 倍的代际飞跃。

## 什么能在哪跑，附例子

能跑和不能跑的代码，具体例子：

```ptx
// ✓ 在 SM 9.0、10.0、12.0 上都能跑——通用
mma.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 ...;

// ✓ 在 SM 9.0、10.0、12.0 上都能跑——但 12.0 上吞吐较低
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 ...;

// ✓ 只能在 SM 10.0 上跑
tcgen05.mma.cta_group::1.kind::nvf4 ...;

// ✓ 只能在 SM 10.0 上跑
tcgen05.mma.cta_group::2.kind::nvf4 ...;

// ✓ 只能在 SM 10.0 上跑
tcgen05.cp.shared::cta::tmem.b64 ...;
```

后三条中的任何一条，用 `--gpu-name=sm_120` 编译时 `ptxas` 都会报错：

```
ptxas fatal: Internal error: instruction 'tcgen05.mma' not supported in this PTX version
```

## 各个库怎么处理

CUTLASS 把选择权交给用户：

```cpp
// CUTLASS Blackwell 数据中心版模板——使用 tcgen05
using GemmKernel = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,                    // ← 架构选择
    ...,
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveOp::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;
```

换成 `cutlass::arch::Sm120` 就会选中另一棵并行的模板树，用 `mma.sync` 和 `wgmma.async`，tile 形状也更小，能装进 99 KiB 的 SMEM 上限。

相比之下，DeepGEMM 目前只有面向 `Sm100` 的代码路径（截至 2026 年初）；`Sm120` 移植在进行中但尚未合入。在工作站版 Blackwell 上加载 DeepGEMM 的 kernel 会在运行时失败。

FlashInfer 有基于 Triton 和基于 CUTLASS 的两套注意力 kernel；CUTLASS-Blackwell 路径用 `tcgen05`，Triton 路径不用，所以工作站版 Blackwell 会回退到 Triton 路径，吞吐有所下降。

## 把 `tcgen05` 翻译成 `mma.sync`

如果你手上有一个只支持 SM100 的 kernel，又需要在 SM120 上跑，概念上的对应关系如下：

| SM100 操作 | SM120 等价物 |
| --- | --- |
| `tcgen05.alloc N` | 分配 N 字节的 `__shared__`（计入 99 KiB） |
| `tcgen05.cp.shared::cta::tmem` | `cp.async.bulk` 或普通的 SMEM 暂存 |
| `tcgen05.mma.cta_group::1.kind::nvf4 m128n128k64` | 约 256 条顺序执行的 `mma.sync m16n8k32`，在寄存器中累加 |
| `tcgen05.commit` | `bar.sync`，或者干脆等最后一条 `mma.sync` 完成 |
| `tcgen05.cp.tmem.shared::cta` | 直接从寄存器存到 SMEM |
| `tcgen05.dealloc` | 作用域结束 |

这种翻译是**机械的**，但产出的 PTX **多得多**——最大的 tile 大约要多 256 倍的指令。最终每 SM 实际达到的 Tensor Core 吞吐大约是 SM120 最优吞吐的 40–70%（而 SM120 的最优吞吐本身又只是 SM100 最优吞吐的一部分）。详细套路见 [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md)。

## 自测

读完你应该能回答：

- TMEM 是什么？它和 SMEM、寄存器有什么区别？
- `wgmma.async` 已经提供了异步 MMA，为什么还要有 `tcgen05.mma`？
- `cta_group::2` 是什么意思？
- 为什么 SM120 没有 `tcgen05`？
- FP4 GEMM 上，SM120 的吞吐和 SM100 大致是什么关系？

## 另见

- [`sm100-vs-sm120`](sm100-vs-sm120.md) —— 完整的架构差异
- [`thread-block-clusters`](thread-block-clusters.md) —— cluster 与 CTA pair MMA
- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) —— `mma.sync` 与 `wgmma.async` 的背景
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) —— 移植套路
- *NVIDIA PTX ISA 8.5*，“TensorCore instructions” → “tcgen05 family”
- *NVIDIA Blackwell Architecture Whitepaper*，“Fifth-Generation Tensor Cores”
