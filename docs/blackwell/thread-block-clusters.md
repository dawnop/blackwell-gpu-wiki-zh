# 线程块簇（thread block cluster）

介于 CTA 和 grid 之间的一级调度单位。Hopper 引入，数据中心版 Blackwell 上扩大了规模，**工作站版 Blackwell 上没有**。它是"能编译、能启动、结果却是错的"这类 kernel 问题中出乎意料常见的一个根源。

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

对 `tcgen05.mma.cta_group::2` 来说，cluster 是**必需的**：最大的 MMA tile（m256n128k64）需要两个 CTA 协作，因为任何一个 CTA 单独都没有足够的 TMEM。

## 各架构的 cluster 大小

| 架构 | 最大 cluster 大小 |
| --- | --- |
| Volta（SM 7.0）– Ampere（SM 8.x） | 1（没有 cluster） |
| Hopper（SM 9.0） | 最多 8（常规），16（显式开启"可移植 cluster 大小"选项后；译注：CUDA 里该属性实为 `cudaFuncAttributeNonPortableClusterSizeAllowed`，即"非可移植"） |
| 数据中心版 Blackwell（SM 10.0） | 最多 16 |
| 工作站版 Blackwell（SM 12.0） | **1（没有 cluster）** |

SM120 这一行就是坑所在：一个带 `.cluster_dim 2,1,1`、为 `sm_120` 编译的 kernel 会：

1. 编译成功（`ptxas` 接受 `.cluster_dim` 指示符）
2. 在设备上加载成功
3. 启动时 cluster 维度**被悄悄降成 (1,1,1)**
4. 如果 kernel 用到 `cluster.sync` 或 cluster 共享 SMEM 寻址，就会**死锁**或**读到垃圾数据**

这是 [`sm100-vs-sm120`](sm100-vs-sm120.md) 里列出的静默失败类型之一。

## 与 cluster 相关的 PTX

### 同步

```ptx
cluster.sync.aligned;            // cluster 内所有 CTA 的屏障
cluster.arrive.aligned %sema;     // 在 cluster mbarrier 上 arrive
cluster.wait.aligned %sema;       // 在 cluster mbarrier 上 wait
```

`cluster.sync` 是 cluster 级屏障。cluster 里所有 CTA 都必须到达；全部到齐后一起继续。在 SM120 上 cluster 大小为 1，`cluster.sync` 就是个空操作（只有一个 CTA，直接往下走），所以靠它做 CTA 间同步的 kernel 会悄无声息地出错。

### 寻址

```ptx
ld.shared::cluster.b32 %r0, [%addr];   // 从同一 cluster 里另一个 CTA 的 SMEM 读
st.shared::cluster.b32 [%addr], %r0;   // 写入另一个 CTA 的 SMEM
```

这些指令跨越同址 SM 的 SMEM。`shared::cluster` 地址空间比单 CTA 的 SMEM 更宽。在 SM120 上，`shared::cluster` 访问会回退到本地 SMEM（因为没有别的 CTA 可访问）——访问一个本该映射到另一个 CTA SMEM 的"cluster 共享"地址，得到的是垃圾数据。

### cluster TMA

```ptx
cp.async.bulk.tensor.shared::cluster.global ...;
```

单条指令把张量 tile 从全局内存异步拷贝到 cluster 共享 SMEM，并分片放到各参与 CTA 里。SM100 支持；SM120 不支持（只有 `cp.async.bulk.tensor.shared::cta`）。

## 检测 kernel 是否用了 cluster

要检查一个预编译 kernel 是否用了大于 1 的 cluster：

```bash
cuobjdump --dump-elf-symbols mylib.so | grep -i cluster

# 或者在导出的 PTX 里找：
cuobjdump --dump-ptx mylib.so | grep -E 'cluster_dim|cluster\.sync|shared::cluster'
```

如果看到 `cluster_dim 2,1,1` 或更大的值，或者任何 `cluster.sync` 指令，说明这个 kernel 依赖 cluster 协作。在 SM120 上跑多半会以不易察觉的方式出错。

## kernel 其实不需要它声明的 cluster 的情况

有些 kernel 声明 `cluster_dim 2,1,1` 只是为了性能（利用 cluster 共享 TMA 的带宽），逻辑上并不需要 CTA 之间协作。这类 kernel 移植到 SM120 是可行的：把 kernel 改成 cluster 大小为 1，用直接的 SMEM 暂存代替 cluster 共享 TMA。会变慢，但结果正确。

CUTLASS 面向 SM100 的模板很多都属于这一类。面向 SM120 的模板存在的意义正是提供不用 cluster 的等价实现。

## kernel 真的需要它声明的 cluster 的情况

用 `tcgen05.mma.cta_group::2` 发射 CTA pair MMA 的 kernel 绝对需要 cluster 大小为 2——m256 这一档的 tile 没有单 CTA 的等价物。依赖这些 tile 的 kernel 必须改写成用较小的 m128 档单 CTA tile（甚至更小的 `mma.sync` tile）。这个改写不是机械替换；tile 形状的选择和分块策略是缠在一起的。

## 一点历史

线程块簇随 Hopper 引入，用来把 Tensor Core 工作扩展到单个 SM 的资源之外。它是一个相当新的编程抽象（2022 年之前没有对应的东西）。Hopper 的 API 通过 `cooperative_groups::cluster_group` 暴露它；CUDA C++ 通过 `__cluster_dims__` 支持它。

消费级 Blackwell *去掉* cluster 这件事很不寻常——NVIDIA 通常会保留已经引入的特性。可能的原因是：cluster 协作需要额外的 SM 到 SM 硬件连接（cluster 共享 SMEM 总线），GB202 芯片为了省面积有意省掉了它。

结果就是：按"Hopper 或更新的架构都支持 cluster"这一假设写的代码，在 SM120 上会意外失败，尽管它的计算能力（12.0）比 Hopper（9.0）*更新*。这是少见的例子——更高的 CC 编号并没有严格包含更低 CC 编号的全部特性，打破了一个通常很可靠的假设。

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
- *NVIDIA PTX ISA 8.5*，关于 `.cluster_dim`、`cluster.sync`、`shared::cluster` 的章节
- *CUDA C++ Programming Guide*，"Thread Block Clusters"一节
