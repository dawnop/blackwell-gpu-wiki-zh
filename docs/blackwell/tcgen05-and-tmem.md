# tcgen05 与 Tensor Memory

仅数据中心版 Blackwell 才有的新 Tensor Core 指令族，以及它所依赖的那种片上存储。

## `tcgen05` 是什么

`tcgen05` 是 PTX ISA 8.6 引入（随 CUDA 12.8 发布，之后的版本继续扩充）的一族 PTX 指令，只面向数据中心版 Blackwell 的架构专属目标（`sm_100a`，以及同家族的 `sm_103a` 等）（译注：原文写 PTX ISA 8.4/8.5，已改）。`5` 指第 5 代 Tensor Core，`gen05` 表示这是第 5 代专属的。它的设计目标：

1. **把 Tensor Core 的执行和 warp 的执行解耦。** warp 发射一条 MMA 后继续往下走；Tensor Core 在旁边并行跑到结束。
2. **支持比 `wgmma.async` 更大的 MMA tile。** 单 CTA 最大 M=128、N=256；CTA pair 最大 M=256、N=256（译注：原文写 128×128 和 256×128，按 PTX ISA 已改）。
3. **减轻寄存器堆的带宽压力。** 累加器放在 TMEM 里，而不是寄存器里。

这几条加起来，在同一个 SM 上，相比基于 `wgmma.async` 的 kernel，FP4/FP6/FP8 峰值吞吐大约能到 **2–3 倍**。

## 指令一览

（译注：原文这张表里的指令拼法和语义大半与 PTX ISA 不符——没有按字节分配、没有 `tcgen05.wait`、也没有 TMEM→SMEM 的 `cp`。下表按 PTX ISA 重写，省略了部分限定符。）

| 指令 | 作用 |
| --- | --- |
| `tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32 [dst], nCols` | 分配 nCols **列** TMEM（32 到 512 之间的 2 的幂），把拿到的 TMEM 地址写进 SMEM 地址 `dst`；整个 warp 一起执行 |
| `tcgen05.dealloc.cta_group::N.sync.aligned.b32 taddr, nCols` | 释放 |
| `tcgen05.relinquish_alloc_permit.cta_group::N.sync.aligned` | 声明本 CTA 不会再分配，好让同一个 SM 上的其他 CTA 拿到 TMEM |
| `tcgen05.ld.sync.aligned.<shape>.x<n>.b32 {regs}, [taddr]` | TMEM → 寄存器（warp 集体执行） |
| `tcgen05.st.sync.aligned.<shape>.x<n>.b32 [taddr], {regs}` | 寄存器 → TMEM |
| `tcgen05.wait::ld.sync.aligned` / `tcgen05.wait::st.sync.aligned` | 等前面的 `ld` / `st` 真正完成 |
| `tcgen05.cp.cta_group::N.<shape> [taddr], sdesc` | SMEM → TMEM（只有这一个方向；常用来把块缩放因子搬进 TMEM） |
| `tcgen05.shift.cta_group::N.down [taddr]` | 把 TMEM 里的数据整体下移 32 个 lane（配合 weight-stationary 的 `tcgen05.mma.ws` 用） |
| `tcgen05.mma.cta_group::N.kind::<kind> [dtmem], adesc, bdesc, idesc, enable_input_d` | MMA：D(TMEM) = A×B (+D)。A、B 通过 SMEM 矩阵描述符给出（A 也可以放在 TMEM），`idesc` 描述形状和数据类型；**由单个线程发射** |
| `tcgen05.mma.cta_group::N.kind::mxf4nvf4.block_scale.scale_vec::4X … [scale_a], [scale_b]` | 带块缩放因子的 MMA，缩放因子放在 TMEM 里 |
| `tcgen05.commit.cta_group::N.mbarrier::arrive::one.shared::cluster.b64 [mbar]` | 把这个线程之前发射的所有 `tcgen05` 异步操作打成一组，全部完成时向 mbarrier 做一次 arrive |
| `tcgen05.fence::before_thread_sync` / `tcgen05.fence::after_thread_sync` | 在 `tcgen05` 异步操作和普通线程同步（`bar.sync`、mbarrier）之间建立先后顺序 |

`<kind>` 列举支持的 MMA 类型：`f16`（FP16/BF16 输入）、`tf32`、`f8f6f4`（FP8/FP6/FP4 混用）、`i8`、`mxf8f6f4`、`mxf4`、`mxf4nvf4`（MXFP4 和 NVFP4，带块缩放）。

## 一段完整的 `tcgen05` MMA PTX

一个简化的数据中心版 Blackwell GEMM tile：

（译注：原文的示例用的是不存在的指令拼法，下面按 PTX ISA 重写，仍是示意，省略了描述符构造和 mbarrier 初始化。）

```ptx
.shared .b32 tmem_slot;          // alloc 会把 TMEM 地址写到这里
.shared .b64 mma_bar;            // mbarrier
.reg .b32 %taddr, %idesc, %r<32>;
.reg .b64 %adesc, %bdesc;
.reg .pred %acc;

// 1. 分配 128 列 TMEM：128 lane × 128 列 × 4 B = 64 KB，正好放一个 m128n128 的 FP32 累加器
//    整个 warp 一起执行；分配完立刻声明不再分配，让别的 CTA 能拿到剩余 TMEM
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem_slot], 128;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
tcgen05.fence::before_thread_sync;
bar.sync 0;                                   // 让其他 warp 看到这个地址
tcgen05.fence::after_thread_sync;
ld.shared.b32 %taddr, [tmem_slot];

// 2. （操作数 A、B 已经由 TMA 暂存到 SMEM；%adesc / %bdesc 是它们的矩阵描述符，
//     %idesc 描述 M/N/K、数据类型和布局）

// 3. 单个线程发射 MMA：D(TMEM) = A × B (+ D)。第一次累加时 %acc 为 false，表示不读旧的 D
tcgen05.mma.cta_group::1.kind::f16 [%taddr], %adesc, %bdesc, %idesc, %acc;

// 4. 把刚才发射的 MMA 打成一组；全部完成时向 mma_bar 做一次 arrive
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mma_bar];

// 5. MMA 运行期间 warp 可以去干别的（比如让 TMA 装下一块数据）

// 6. 等 mbarrier（普通的 mbarrier.try_wait 循环，这里省略）

// 7. 把累加器从 TMEM 读进寄存器（每个 warp 只能读自己那 32 个 lane），再写到 SMEM 或全局内存
tcgen05.ld.sync.aligned.32x32b.x32.b32 {%r0, ..., %r31}, [%taddr];
tcgen05.wait::ld.sync.aligned;

// 8. 释放 TMEM
tcgen05.dealloc.cta_group::1.sync.aligned.b32 %taddr, 128;
```

和 `wgmma.async` 对比：

- `wgmma` 的累加器在**寄存器**里；`tcgen05` 的累加器在 **TMEM** 里
- `wgmma` 由一个 **warp group**（4 个 warp）发射；`tcgen05.mma` 由**单个线程**发射（`alloc` / `ld` / `st` 则是单个 warp 集体执行）
- `wgmma` 的 tile 最大到 m64n256k16；`tcgen05` 能到 M=128、N=256（单 CTA）或 M=256、N=256（CTA pair），K 随数据宽度变（FP16 是 16，FP8 是 32，FP4 是 64）
- 两者都是异步的：`wgmma` 用 `commit_group` / `wait_group`，`tcgen05` 用 `commit` 把完成挂到 mbarrier 上

## Tensor Memory（TMEM）

一种新的片上存储。特性：

- **容量**：每个 SM 256 KB，组织成 128 个 lane × 512 列，每格 32 位
- **分配粒度**：按列分配，一次 32 到 512 列之间的 2 的幂（32 列 = 16 KB）（译注：原文写"128 字节"，已改）
- **分配器**：`tcgen05.alloc` 返回 TMEM 基地址，`tcgen05.dealloc` 释放
- **寻址**：TMEM 地址与 SMEM 和全局地址**相互独立**——它是一个 32 位逻辑地址，高 16 位是 lane、低 16 位是列
- **带宽**：足以让 `tcgen05.mma` 以峰值速率运行
- **可见性**：TMEM 是每个 SM 私有的；发射指令的 CTA（或 CTA pair）可以寻址它，其他 CTA 不行

TMEM 的存在只为一个原因：在 FP4/FP6 MMA 的吞吐量级上，对 `wgmma.async` 风格的 kernel 来说，**寄存器堆的带宽成了瓶颈**。Tensor Core 消耗操作数的速度，超过了一个 warp 的寄存器读取所能供给的速度。把累加器挪出寄存器（把操作数暂存也挪出寄存器，先进 SMEM 再进 TMEM）之后，warp 的寄存器堆就只需要服务**发射和收尾**两个阶段，而不用伺候运行中的 MMA。

一个好用的心智模型：TMEM 之于 Tensor Core，就像 L1 缓存之于 ALU。

### TMEM 的组织方式

（译注：原文列出的"默认 / 步长 / 复制 / 压缩"四种布局以及"`tcgen05.shift` 做布局变换"在 PTX ISA 里并不存在，本节按 ISA 改写。）

TMEM 是一个 128 lane × 512 列的二维阵列。累加器 D 的第 i 行落在第 i 个 lane 上（M=128 时正好占满，M=64 时只占一半），N 就是占用的列数（FP32 累加时每列存一个值）。所以一个 m128n256 的 FP32 累加器占 256 列，也就是半个 TMEM。

`tcgen05.ld` / `tcgen05.st` 按几种固定形状（`32x32b`、`16x64b`、`16x128b`、`16x256b`）搬数据，而且一个 warp 只能访问它在 warpgroup 里对应的那 32 个 lane（warp 0 管 lane 0–31，warp 1 管 lane 32–63，以此类推）。这就是为什么 epilogue 一定要 4 个 warp 一起做。

`tcgen05.shift.down` 把一段 TMEM 里的数据整体下移 32 个 lane，配合 weight-stationary 形式的 `tcgen05.mma.ws` 使用，一般 GEMM 用不到。

## CTA pair / `cta_group::2` 模式

M=256 的 `tcgen05.mma` tile 一个 CTA 装不下：它的 TMEM 只有 128 个 lane。`cta_group::2` 模式让两个 CTA 协作（译注：原文说"两个 CTA 通过 cluster 的 SMEM 链路共享 TMEM 分配、步调一致地各发一条"，按 PTX ISA 改写如下）：

- 两个 CTA 作为同一个 **cluster** 的成员启动（cluster 维度为 2）
- 两个 CTA 的 SMEM 和 TMEM 都参与：A、B 从两个 CTA 的 SMEM 里各取一半，结果 D 的 256 行分别落到两个 CTA 的 TMEM（各 128 个 lane）
- MMA 只由其中一个 CTA（leader）的一个线程发射一次；`commit` 可以让完成信号同时到达两个 CTA 里的 mbarrier

这是 SM100 支持大于 1 的线程块簇的原因之一：`tcgen05` 的 CTA pair 模式需要它。

**工作站版 Blackwell 有 cluster，但没有 `tcgen05`，所以也没有 CTA pair MMA。**（译注：原文写"不支持大于 1 的 cluster"，与 CUDA 编程指南不符，已改。） 为 SM120 编译的 kernel 只能用单 CTA 的 tile 形状——或者更常见的情况是，根本不用 `tcgen05`。

## 为什么工作站版 Blackwell 没有 `tcgen05`

NVIDIA 大概的考虑（从架构反推）：

- TMEM 很占芯片面积（每 SM 256 KB 是实打实的硅片）
- cluster 执行需要额外的 SM 到 SM 链路（又是硅片）
- 消费级负载（游戏、内容创作、轻量 ML）从 m128n256k64 这种大 tile 的 GEMM 里得不到多少好处
- 把数据中心和消费级区分开是有意为之的产品策略

结果就是：工作站版 Blackwell 有着**同样的 Tensor Core 硬件**（第 5 代，原生 FP4/FP6/FP8），但只能通过 `mma.sync`（以及 `sm_120a` 专属的块缩放 `mma.sync`）访问，而它受寄存器限制（译注：原文写"`mma.sync` 和 `wgmma.async`"，SM120 没有 `wgmma`）。所以它每 SM 的 FP4 峰值吞吐和 Hopper 每 SM 的 FP8 吞吐差不多——有用，但没有 SM100 那种 2–3 倍的代际飞跃。

## 什么能在哪跑，附例子

能跑和不能跑的代码，具体例子：

```ptx
// ✓ 在 SM 9.0、10.0、12.0 上都能跑——通用
mma.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 ...;

// ✓ 只能在 SM 9.0（sm_90a）上跑——Blackwell 两个分支都没有 wgmma（译注：原文说 10.0、12.0 也能跑，已改）
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 ...;

// ✓ 只能在 SM 10.0（sm_100a）上跑
tcgen05.mma.cta_group::1.kind::mxf4nvf4 ...;

// ✓ 只能在 SM 10.0（sm_100a）上跑
tcgen05.mma.cta_group::2.kind::mxf4nvf4 ...;

// ✓ 只能在 SM 10.0（sm_100a）上跑
tcgen05.ld.sync.aligned.32x32b.x32.b32 ...;
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

换成 `cutlass::arch::Sm120` 就会选中另一棵并行的模板树，用 `mma.sync`（译注：原文还写了 `wgmma.async`，SM120 没有它），tile 形状也更小，能装进 99 KiB 的 SMEM 上限。

相比之下，DeepGEMM 目前只有面向 `Sm100` 的代码路径（截至 2026 年初）；`Sm120` 移植在进行中但尚未合入。在工作站版 Blackwell 上加载 DeepGEMM 的 kernel 会在运行时失败。

FlashInfer 有基于 Triton 和基于 CUTLASS 的两套注意力 kernel；CUTLASS-Blackwell 路径用 `tcgen05`，Triton 路径不用，所以工作站版 Blackwell 会回退到 Triton 路径，吞吐有所下降。

## 把 `tcgen05` 翻译成 `mma.sync`

如果你手上有一个只支持 SM100 的 kernel，又需要在 SM120 上跑，概念上的对应关系如下：

| SM100 操作 | SM120 等价物 |
| --- | --- |
| `tcgen05.alloc nCols` | 把累加器放进寄存器，放不下的部分放 `__shared__`（计入 99 KiB） |
| `tcgen05.cp`（SMEM → TMEM 的缩放因子） | 直接从 SMEM 读进寄存器 |
| `tcgen05.mma.cta_group::1.kind::mxf4nvf4 m128n128k64` | 128 条 `mma.sync m16n8k64`（`kind::mxf4nvf4.block_scale`），分摊到 4 个 warp 各 32 条，在寄存器中累加 |
| `tcgen05.commit` + mbarrier | 什么都不用做——`mma.sync` 是同步的，返回即完成 |
| `tcgen05.ld`（TMEM → 寄存器） | 结果本来就在寄存器里 |
| `tcgen05.dealloc` | 作用域结束 |

（译注：原文这张表沿用了不存在的指令拼法和"约 256 条 m16n8k32"的数字，已按 PTX ISA 改写；SM120 的 FP4 `mma.sync` 形状是 m16n8k64。）

这种翻译是**机械的**，但产出的 PTX **多得多**——最大的单 CTA tile（m128n256k64）要拆成 256 条 `mma.sync`。最终每 SM 实际达到的 Tensor Core 吞吐大约是 SM120 最优吞吐的 40–70%（而 SM120 的最优吞吐本身又只是 SM100 最优吞吐的一部分）。详细套路见 [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md)。

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
