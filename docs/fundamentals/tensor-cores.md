# Tensor Core

每个 SM 内部的专用执行单元，一条指令就能对一小块 tile 完成矩阵乘累加（MMA）。GPU 之所以适合跑 transformer 推理，原因就在这里。

## 它做什么，为什么要有它

一条普通的 warp 级指令并行处理 32 个 32 位元素——32 个 SIMT lane，每个 lane 一个元素。而一条 Tensor Core 指令处理的是一个**矩阵 tile**：一小块 16×8 或 16×16 的元素，一条指令就算完这个 tile 的整个 `D = A · B + C`。

吞吐提升非常可观。以 Hopper 上 BF16 输入、FP32 累加器为例：

- 普通 FP32 FMA：约 30 TFLOPs/GPU
- BF16 Tensor Core MMA：约 990 TFLOPs/GPU

差不多 30 倍。数据中心版 Blackwell 上跑 FP4，加速比更大（相对普通 FP32 约 165 倍）。

对程序员来说，Tensor Core 本身是不可见的，只表现为一条特殊指令。它没有单独的"Tensor Core 内存"（好吧，现在有了——TMEM，但那是 SM100 的东西，见下文）——操作数来自寄存器（或 SMEM、或 TMEM），结果落到寄存器（或 TMEM）。

## 五代 Tensor Core

| 代 | 架构 | 计算能力 | 关键特性 |
| --- | --- | --- | --- |
| 1 | Volta | 7.0（V100） | FP16 输入，FP32 累加。引入 `mma.sync`。 |
| 2 | Turing | 7.5（T4、RTX 20） | + INT8、INT4、INT1 |
| 3 | Ampere | 8.0–8.9（A100、RTX 30） | + BF16、TF32。`ldmatrix` 用于快速加载 SMEM。 |
| 4 | Hopper | 9.0（H100/H200） | + FP8（E4M3、E5M2）。`wgmma.async`（warp 组异步 MMA）。TMA。线程块簇。 |
| 5 | Blackwell | 10.0 / 12.0 | + FP6、FP4、MX-FP4、NVFP4。`tcgen05` 指令族（仅 SM100）。TMEM（仅 SM100）。 |

本 wiki 主要讲第 5 代。前几代提一下是为了铺垫背景——你会在现代 kernel 代码里看到 `mma.sync` 和 `wgmma.async`，尤其是作为回退路径。

## MMA 指令族

三个相关的指令族。它们都是 PTX 层面的东西；你很少在 CUDA C++ 里直接写它们，但在编译出来的 PTX 和 CUTLASS 模板里会经常见到。

### `mma.sync`——通用，自 Volta 起

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  %rd0, %rd1, %rd2, %rd3;
//                ^^^^^^^^^         ^^^^ ^^^^ ^^^^
//                tile 形状          A     B    C/D
//                                  类型  类型  类型
```

读作："同步地对一个 m16n8k16 的 tile（16 行、8 列、内积深度 16）做 MMA，A 行主序、B 列主序，A 和 B 是 BF16，累加器是 FP32，输出在寄存器 `%rd0`，操作数在 `%rd1`、`%rd2`、`%rd3`。"

特点：

- **同步**：warp 发射这条指令，**warp 里全部 32 个线程都必须参与**。结果立刻就在寄存器里。
- **tile 小**：m16n8k16（BF16/FP16）、m16n8k32（FP8/FP4）。更大的逻辑 tile 要拆成多条 `mma.sync` 依次发射。
- **通用**：从 Volta 起每一代 NVIDIA GPU 都支持。**SM100 和 SM120 都能跑。**

从 Volta 到 Ampere，它差不多是所有 Tensor Core 代码的主力，也是每一代架构上的回退路径。实践中看到的大多数 PTX 里都有 `mma.sync`。

### `wgmma.async`——Hopper 引入

```ptx
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16  %rd0, %rd1, %rd2, %rd3;
```

特点：

- **异步**：warp 发射指令后继续干别的活；结果稍后才落地。通过 `wgmma.commit_group.sync` 和 `wgmma.wait_group.sync` 同步。
- **warp 组**：一个 *warp 组*是 4 个 warp（128 线程）。MMA 由整个 warp 组发射，而不是单个 warp。
- **tile 更大**：从 m64n128k16 到 m64n256k16，而 mma.sync 只有 m16n8k16。同样的逻辑工作量，发射的指令更少。
- **只有 Hopper**：`wgmma` 是 `sm_90a` 专属指令。数据中心版 Blackwell 用 `tcgen05.mma` 取代了它，工作站版 Blackwell 则只有 `mma.sync`。**Blackwell 两个分支都不能跑 `wgmma`。**（译注：原文称 SM 10.0 和 SM 12.0 也能跑，与 PTX ISA 不符，已改。）

`wgmma.async` 开启了"warp 组上一切皆异步"的时代。现代 Hopper kernel（FA-3、CUTLASS 的 Hopper 模板）都重度依赖它。

### `tcgen05.mma`——仅限数据中心版 Blackwell

```ptx
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
    [%tmem_d], %a_desc, %b_desc, %idesc, [%tmem_scale_a], [%tmem_scale_b], %acc;
//          ^^^^^^^^^^^^ ^^^^^^^^^^^^^^
//          单 CTA        FP4 输入，带块缩放因子
```

特点：

- **异步且解耦**：比 `wgmma` 更彻底。A、B 通过 SMEM 矩阵描述符给出（A 也可以放在 TMEM），累加器和块缩放因子都在 **TMEM** 里，寄存器完全不参与。整条指令由**单个线程**发射，几乎不花时间。
- **tile 更大**：单 CTA 最大 M=128、N=256；CTA pair（用 `cta_group::2`）最大 M=256、N=256（译注：原文写 m128n128k64 / m256n128k64，按 PTX ISA 已改）。
- **CTA pair / 双 CTA cluster 模式**：两个 CTA 各出一半 SMEM 里的操作数、各收一半 TMEM 里的结果，对一个更大的 tile 只发射一条 MMA。需要 cluster 维度为 2。**仅 SM100。**
- **配套指令**：`tcgen05.alloc` 分配 TMEM，`tcgen05.commit` 把完成挂到 mbarrier 上，`tcgen05.ld` / `tcgen05.st` 在 TMEM 和寄存器之间搬数据，`tcgen05.cp` 从 SMEM 拷进 TMEM。
- **仅 SM100。** **SM120 上不能用。**

`tcgen05` 之所以存在，是因为到了 FP4/FP6 这种吞吐水平，warp 组 MMA 的做法开始卡在寄存器堆带宽上——warp 发射 MMA 指令的速度跟不上 Tensor Core 的消耗速度。把累加器从寄存器搬到 TMEM 之后，warp 发射一条 `tcgen05.mma`，Tensor Core 就在 TMEM 里一路算完，warp 可以同时去干别的。

正因为这种深度耦合，`tcgen05` *不是*简单地往 ISA 里加几条指令——围绕它的整套 kernel 设计模式（TMEM 分配、异步 commit、TMA 直接拷进 TMEM）自成一个生态。

## 什么能跑在什么上

| 指令族 | SM 7.x | SM 8.x | SM 9.0 | SM 10.0 | SM 12.0 |
| --- | :---: | :---: | :---: | :---: | :---: |
| `mma.sync` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `wgmma.async` | | | ✓ | | |
| `tcgen05.mma`（单 CTA） | | | | ✓ | |
| `tcgen05.mma`（CTA pair） | | | | ✓ | |
| 块缩放 `mma.sync`（`kind::mxf4nvf4` 等） | | | | | ✓ |

（译注：原文表里 `wgmma.async` 在 SM 10.0 和 12.0 两列都打了勾，与 PTX ISA 不符，已改；另补上 SM 12.0 专属的块缩放 `mma.sync` 一行。）

## CUTLASS 怎么选

CUTLASS 作为高性能 GEMM 的参考库，为每个 MMA 指令族维护了各自独立的模板体系：

- `cutlass/include/cutlass/gemm/collective/sm80_*`——Ampere，基于 `mma.sync`
- `cutlass/include/cutlass/gemm/collective/sm90_*`——Hopper，基于 `wgmma`
- `cutlass/include/cutlass/gemm/collective/sm100_*`——数据中心版 Blackwell，基于 `tcgen05`
- `cutlass/include/cutlass/gemm/collective/sm120_*`——工作站版 Blackwell，基于 `mma.sync`（含块缩放版）

实例化一个 CUTLASS GEMM 时，你指定目标架构，模板就会选对应的 MMA 指令族。SM100 模板的目标是 **`sm_100a`**，因为它们需要 `tcgen05`。SM120 模板的目标是 **`sm_120a`**（块缩放 `mma.sync` 是 `a` 专属）或 **`sm_120` / `sm_120f`**，走的是更老但依然有效的 `mma.sync` 路径（译注：原文写 `mma.sync` / `wgmma`，SM120 没有 `wgmma`）。

这就是为什么用 CUTLASS 针对一个目标编出来的库，在另一个目标上跑不了：二进制里的指令本身就不一样。

## 性能视角

FP4 精度下，一条 m128n256k64 的 `tcgen05.mma` 含 128×256×64 ≈ 210 万次乘累加，Tensor Core 要花很多个周期才能算完，并不是"每周期一条"。按 B200 公开的 FP4 稠密峰值（9 PFLOPS）、148 个 SM、2.1 GHz 反推，每个 SM 每周期大约 1.4 万次 FP4 乘累加（每个 SM 4 个 Tensor Core，各约 3.6 千次）。（译注：原文写"每个 Tensor Core 每周期发射一条 m128n128k64，即 1,048,576 次乘累加"，数量级差了两百多倍；"B100 144 个 SM、约 5 PFLOPs"也与公开规格不符，已按 B200 的公开数字改写。）

在 SM120 上，没有 `tcgen05`，同样的工作要退回去发射一大堆 `mma.sync m16n8k32`。单个 Tensor Core 的算术吞吐是差不多的——Tensor Core 硬件本身相同——但*调度开销*（发射的指令更多、寄存器堆流量更大）拉低了实际能达到的吞吐。

一个粗略的经验法则：SM120 上的 GEMM kernel **每 FLOP 只能达到 SM100 峰值的 40–70 %**。再看硬件本身：按 NVIDIA 公开规格，RTX PRO 6000 Blackwell 的 FP4 峰值约 4 PFLOPS（稀疏）、约 2 PFLOPS（稠密），B200 约 18 / 9 PFLOPS——**绝对差距约 4–5 倍**。而 GB202 的 SM 数（188 个）比 B200（148 个）还多、频率也更高（2.6 GHz 对 2.1 GHz），折算到每个 SM 每周期，差距约 7 倍——这一部分才是 Tensor Core 数据通路和 ISA（有没有 `tcgen05` / TMEM）造成的。（译注：原文写"RTX PRO 6000 约 125 TFLOPs、B100 约 5 PFLOPs、相差 40 倍、一半来自硬件一半来自 ISA"，与公开规格不符，已改。）

## 常见的 tile 形状

PTX ISA 规定了允许的 tile 形状；各个库从中挑选组合来实例化。常见的有：

| 指令族 | tile 形状 | 场景 |
| --- | --- | --- |
| `mma.sync` | m16n8k16 | FP16/BF16 |
| `mma.sync` | m16n8k32 | FP8/FP4 |
| `wgmma.async` | m64n{16…256}k{16,32} | Hopper FP16/FP8 |
| `tcgen05.mma` | m{64,128}n{64,128}k{16,32,64} | 数据中心版 Blackwell，单 CTA |
| `tcgen05.mma` | m{128,256}n{64,128}k{16,32,64} | 数据中心版 Blackwell，CTA pair |

tile 形状的选择是一种权衡：tile 越大，越能摊薄指令发射开销；tile 越小，占用率越高。

## 自测

你应该能回答：

- `m16n8k16` 是什么意思？
- `mma.sync` 和 `wgmma.async` 有什么区别？
- `tcgen05.mma` 为什么把累加器放在 TMEM 而不是寄存器里？
- CUTLASS 的数据中心版 Blackwell 模板用的目标架构是什么？
- FP4 精度下 SM100 和 SM120 的吞吐差距大概有多大？

## 另见

- [`memory-hierarchy`](memory-hierarchy.md)——Tensor Memory 在存储层次里的位置
- [`number-formats`](number-formats.md)——FP4/FP6/FP8/BF16 详解
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md)——数据中心版 Blackwell 的完整故事
- NVIDIA *PTX ISA* 规范中 MMA / WGMMA / TCGEN05 相关章节
- *CUTLASS* 文档，尤其是"Blackwell architecture"一节
