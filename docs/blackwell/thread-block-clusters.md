# 线程块簇（thread block cluster）

介于 CTA 和 grid 之间的一级调度单位。Hopper 引入，数据中心版 Blackwell 上扩大了规模，工作站版 Blackwell 也有，只是少了建在 cluster 之上的两样 SM100 专属功能：CTA pair MMA 和硬件加速的 multicast TMA。

## cluster 是什么

一个 **cluster** 是一组 CTA，它们：

- 被**协同调度**到一组共享 *cluster 共享内存* 地址空间的 SM 上
- 可以通过 `cluster.sync` 互相**同步**
- 可以通过 cluster 共享寻址**访问**彼此的 SMEM
- 可以发射 **cluster 级 TMA**，一条操作就把一个张量 tile 分发到所有参与 CTA 的 SMEM 里

cluster 大小在 kernel 启动时声明：

```cpp
__global__ __cluster_dims__(2, 1, 1) void my_kernel(...) { ... }
```

或者在 PTX 里：

```ptx
.cluster_dim 2,1,1
```

启动时的 cluster 形状决定有多少个 CTA 参与。cluster 里的每个 CTA 都有一个 `clusterIdx`，可以用 cluster 内的相对索引访问其他 CTA。

## 为什么需要 cluster

对现代 Tensor Core kernel 来说，单个 CTA 的 SMEM（Hopper / SM100 上是 228 KiB）有时装不下这些东西：

- 操作数 A 的暂存区（多级流水线）
- 操作数 B 的暂存区（多级流水线）
- 累加器（大 tile 时很大）
- 流水线状态（邮箱、mbarrier）

cluster 让两个或更多 CTA 为同一个逻辑 kernel tile **合并各自的 SMEM**。有了 cluster 共享寻址，CTA-0 发的 TMA 可以把操作数 A 直接放进 CTA-1 的 SMEM——不需要经过全局内存中转。

对 `tcgen05.mma.cta_group::2` 来说，cluster 是**必需的**：M=256 的 MMA tile 需要两个 CTA 协作，因为一个 CTA 的 TMEM 只有 128 个 lane，装不下 256 行累加器。

## 各架构的 cluster 大小

| 架构 | 最大 cluster 大小 |
| --- | --- |
| Volta（SM 7.0）– Ampere（SM 8.x） | 1（没有 cluster） |
| Hopper（SM 9.0） | 最多 8（可移植上限），16（给 kernel 显式打开 `cudaFuncAttributeNonPortableClusterSizeAllowed` 属性后） |
| 数据中心版 Blackwell（SM 10.0） | 最多 16 |
| 工作站版 Blackwell（SM 12.0） | 最多 8（可移植上限；没有 16 的非可移植选项） |

真正的坑不在 cluster 本身，而在 cluster 之上的 SM100 专属功能。一个带 `.cluster_dim 2,1,1`、为 `sm_120` 编译的 kernel：

1. 编译成功，启动成功，而且确实以 2 个协作 CTA 运行，`cluster.sync` 和 `shared::cluster` 寻址都正常
2. 如果它接着发 `tcgen05.mma.cta_group::2`，`ptxas` 早在编译期就会拒绝
3. 如果它用 multicast TMA，功能上能跑，但没有硬件加速，会明显变慢

所以 SM120 上和 cluster 有关的失败要么是编译期错误（`cta_group::2`），要么是性能问题（multicast），不是静默算错。

## 与 cluster 相关的 PTX

### 同步

```ptx
cluster.sync.aligned;      // cluster 内所有 CTA 的屏障
cluster.arrive.aligned %sema;   // 在 cluster mbarrier 上 arrive
cluster.wait.aligned %sema;    // 在 cluster mbarrier 上 wait
```

`cluster.sync` 是 cluster 级屏障。cluster 里所有 CTA 都必须到达；全部到齐后一起继续。SM120 上它和 Hopper 一样正常工作。

### 寻址

```ptx
ld.shared::cluster.b32 %r0, [%addr];  // 从同一 cluster 里另一个 CTA 的 SMEM 读
st.shared::cluster.b32 [%addr], %r0;  // 写入另一个 CTA 的 SMEM
```

这些指令跨越同址 SM 的 SMEM。`shared::cluster` 地址空间比单 CTA 的 SMEM 更宽。SM120 同样支持分布式共享内存，`shared::cluster` 访问正常。

### cluster TMA

```ptx
cp.async.bulk.tensor.shared::cluster.global ...;
```

单条指令把张量 tile 从全局内存异步拷贝到 cluster 里多个 CTA 的 SMEM。SM100 有硬件多播。SM120 上 PTX 允许写 `.multicast::cluster`，但 PTX ISA 注明这个限定符只针对 sm_90a / sm_100a 系列做了优化、在其他目标上性能会大幅下降，实际等于没有硬件多播。

## SM100 在 cluster 和异步拷贝上新增的东西

Hopper 的那一套（`cp.async.bulk`、`cp.async.bulk.tensor`、mbarrier 全家、`fence.proxy.async`、DSMEM / `mapa`、`__cluster_dims__`、`cudaLaunchKernelEx` 的 cluster 属性、128B swizzle、`.multicast::cluster`）在 `sm_100a` 上原样保留。cluster 上限也没变：可移植 8，B200 可以打开非可移植的 16。在这之上 SM100 加了：

- **TMA 的 `.cta_group::1 / ::2` 限定符**：让 TMA 完成信号打到 CTA pair 里对方 CTA 的 mbarrier。`cta_group::2` 的 MMA 就靠它让 leader 知道两边的数据都到了。
- **新的 TMA 模式**：`.tile::gather4`（4 行合成一个 tile）/ `.tile::scatter4`，以及 `.im2col::w / ::w::128`；driver 端新增 `cuTensorMapEncodeIm2colWide`。
- **128B swizzle 的原子性可配**：16B、32B、32B + 8B 翻转、64B（`CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B / _ATOM_32B_FLIP_8B / _ATOM_64B`）。配套新增 FP4 / FP6 解包数据类型 `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B / 16U4_ALIGN16B / 16U6_ALIGN16B`。
- **`st.bulk`**：SMEM 批量清零（最多 16 MiB，初值只能是 0）。
- **`clusterlaunchcontrol.try_cancel / query_cancel`**：让正在跑的 cluster 把还没启动的 cluster 的工作"抢"过来，做动态调度。CUTLASS 的 SM100 persistent tile scheduler（`PersistentTileSchedulerSm100` + `PipelineCLCFetchAsync`）就是靠它。
- **`setmaxnreg` 保留**，`sm_100a` / `sm_100f` 都支持，语义不变（24 到 256，8 的倍数，整个 warpgroup 一致）。
- `st.async` / `red.async` 能打到 `.global` 并带 `.release / .scope / .mmio`（PTX 8.7）。

对写 kernel 的人来说，最要紧的是前两条之一：`cta_group::2` 要求 cluster 的 CTA 总数为偶数，TMA 完成信号必须带 `.cta_group::2` 才能通知到对方。

## 检测 kernel 是否用了 cluster

要检查一个预编译 kernel 是否用了大于 1 的 cluster：

```bash
cuobjdump --dump-elf-symbols mylib.so | grep -i cluster

# 或者在导出的 PTX 里找：
cuobjdump --dump-ptx mylib.so | grep -E 'cluster_dim|cluster\.sync|shared::cluster'
```

如果看到 `cluster_dim 2,1,1` 或更大的值，或者任何 `cluster.sync` 指令，说明这个 kernel 依赖 cluster 协作。在 SM120 上，只有其中的 `tcgen05.mma.cta_group::2` 会真正失败（编译期），multicast 只是变慢。

## kernel 其实不需要它声明的 cluster 的情况

有些 kernel 声明 `cluster_dim 2,1,1` 只是为了性能（利用 cluster 共享 TMA 的带宽），逻辑上并不需要 CTA 之间协作。这类 kernel 移植到 SM120 是可行的：把 multicast TMA 换成每个 CTA 各自的 TMA（cluster 本身可以留着）。会变慢，但结果正确。

CUTLASS 面向 SM100 的模板很多都属于这一类。面向 SM120 的模板存在的意义正是提供不用 cluster 的等价实现。

## kernel 真的需要它声明的 cluster 的情况

用 `tcgen05.mma.cta_group::2` 发射 CTA pair MMA 的 kernel 绝对需要 cluster 大小为 2——m256 这一档的 tile 没有单 CTA 的等价物。依赖这些 tile 的 kernel 必须改写成用较小的 m128 档单 CTA tile（甚至更小的 `mma.sync` tile）。这个改写不是机械替换；tile 形状的选择和分块策略是缠在一起的。

## 一点历史

线程块簇随 Hopper 引入，用来把 Tensor Core 工作扩展到单个 SM 的资源之外。它是一个相当新的编程抽象（2022 年之前没有对应的东西）。Hopper 的 API 通过 `cooperative_groups::cluster_group` 暴露它；CUDA C++ 通过 `__cluster_dims__` 支持它。

SM120 相对 Hopper 真正去掉或没有的是 TMEM / `tcgen05`、硬件 multicast 和 228 KB 一级的 SMEM。更高的计算能力编号并不包含更低编号的全部特性，`wgmma` 就是例子：它只在 `sm_90a` 上有，10.x 和 12.x 都没有。

## 自测

你应该能回答：

- cluster 是什么？它和 grid、CTA 有什么区别？
- SM100 上最大 cluster 是多少？SM120 上呢？
- 在 SM120 上启动一个 `cluster_dim 2,1,1` 的 kernel 会发生什么？
- 为什么 `tcgen05.mma.cta_group::2` 需要 cluster？
- 怎么从编译好的二进制判断一个 kernel 用没用 cluster？

## 另见

- [`tcgen05-and-tmem`](tcgen05-and-tmem.md) —— `tcgen05.mma.cta_group::2` 与 CTA pair 执行
- [`sm100-vs-sm120`](sm100-vs-sm120.md) —— 更全面的架构差异
- [`compatibility/cluster-rewriting`](../compatibility/cluster-rewriting.md) —— 移植套路
- *NVIDIA PTX ISA*（9.3），关于 `.cluster_dim`、`cluster.sync`、`shared::cluster` 的章节
- *CUDA C++ Programming Guide*，"Thread Block Clusters"一节
