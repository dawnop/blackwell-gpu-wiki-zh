# cluster 改写

kernel 假定 cluster 大小 > 1，却要跑在 cluster 不可用的硬件上，该怎么办。

## 问题是什么

线程块簇（cluster，见 [`blackwell/thread-block-clusters`](../blackwell/thread-block-clusters.md)）把多个 CTA 编成一组，组内可以通过分布式共享内存互相访问 SMEM。在 SM100 上，大小 2–8 的 cluster 是家常便饭。在 SM120 上，cluster 和分布式共享 SMEM 本身是有的（最多 8），真正没有的是 **CTA pair MMA（`cta_group::2`）和硬件加速的 multicast TMA**（译注：原文称 SM120 唯一安全的 cluster 大小是 1、没有 cluster 级共享 SMEM，与 CUDA 编程指南不符，已改）。

如果 kernel 依赖这两项，就必须改写。下面几种思路仍然适用，只是动机从"没有 cluster"变成"没有 pair MMA / multicast"。有四种思路（译注：原文写"三种"，下文实际列了四种）。

## 思路 1：塌缩成单 CTA

如果 cluster 只是**图方便**（比如把几个 SM 拼在一起凑出更大的虚拟 SMEM 池），往往直接把 `cluster_dim = 1`，接受每个 CTA 的 tile 变小就行。

```cuda
// 原版 SM100
__cluster_dims__(2, 1, 1)
__global__ void kernel(...) {
    // 每个 CTA 处理半个 tile，
    // 通过分布式共享内存访问邻居的 SMEM
    auto neighbor_smem = cg::cluster_group::block_index({1, 0, 0});
    ...
}

// SM120 改写：每个 CTA 独立完成整个半 tile
__global__ void kernel(...) {
    // cluster_dim 隐含为 1
    // 没有跨 CTA 的 SMEM 访问
    ...
}
```

每个 CTA 的 tile 变小了，可能要用下面的办法补回来：

- 多启动一些 CTA（总工作量不变，并发 CTA 更多）
- 借 SM120 更多的 SM 数量来吃掉多出来的 CTA

cluster **只是方便、并非必需**时，这招管用。

## 思路 2：拆成独立的 kernel

如果 cluster 是用来做**协作计算**的——比如一条 `tcgen05.mma.cta_group::2`，跨两个 CTA 发射一条 2 倍宽的 MMA——那就不能简单塌缩。两个 CTA 做的是实打实不同的工作，合起来才是一个输出 tile。

改法：把工作拆成两个独立 kernel（或同一 kernel 的两个 CTA），各自产出一半输出，再在更上层把结果合起来。

```cuda
// 原版：m256n128 协作 MMA
__cluster_dims__(2, 1, 1)
__global__ void mma_cluster_pair(...) {
    if (cluster_block_id() == 0) {
        // 通过 tcgen05.mma.cta_group::2 产出上半部分
    } else {
        // 用下半部分的数据为上半部分做贡献
    }
}

// SM120 改写：两条独立的 m128n128 MMA
__global__ void mma_independent(int half_id, ...) {
    if (half_id == 0) {
        // 用单 CTA mma 产出上面的 m128n128 tile
    } else {
        // 用单 CTA mma 产出下面的 m128n128 tile
    }
    // 调用方分别以 half_id=0 和 half_id=1 启动
}
```

输出现在变成两块，下游消费者必须两块都读。改动范围大，但机械、不费脑子。

## 思路 3：用全局内存模拟 cluster 共享

如果 cluster 大量使用分布式共享内存的访问模式（比如 stencil 计算，每个 CTA 要读很多邻居的数据），最干净的办法可能是改用经 L2 的全局内存。

```cuda
// 原版 SM100：读邻居的 SMEM
auto neighbor = cg::cluster_group::block_index({1, 0, 0});
float val = neighbor.shared_buffer[idx];

// SM120 改写：每个 CTA 把自己那份写到全局内存，
// 再从邻居的全局内存区域读回来
__shared__ float local_buf[N];
// ... 计算 local_buf ...
__syncthreads();
// 把 local_buf 写到本 CTA 的全局内存区域
gmem_buf[my_block_id * N + threadIdx.x] = local_buf[threadIdx.x];
__threadfence();
// 从全局内存读邻居的区域
float val = gmem_buf[neighbor_block_id * N + idx];
```

性能代价：全局内存访问比分布式共享内存慢得多。但 L2（工作站 Blackwell 上有 96 MB）够大，数据往往能留在缓存里，损失没那么惨。

## 思路 4：不改写，直接换

对某些 kernel（尤其是 CUTLASS 这类库里的），最干净的修法是换一套**直接面向 SM120 的模板**。CUTLASS 既有 SM100 模板（用 cluster 和 tcgen05），也有 SM120 模板（单 CTA、mma.sync）。如果你的 kernel 用的是 CUTLASS，把它配置成分发到 SM120 模板树就行，什么都不用改写。

只要用得上，这是最干净的路，没有之一。

## 怎么发现

怎么知道一个 kernel 用了 cluster？找这些：

```cuda
__cluster_dims__(X, Y, Z)         // CUDA C++ 属性
.cluster_dim X, Y, Z;              // PTX 指示
cooperative_groups::cluster_group  // C++ API
__cluster_size_in_blocks           // 内建函数
distributed_shared_memory_address  // 分布式共享内存访问
cg::cluster_barrier                // cluster 范围的 barrier
```

在已编译的二进制里，去 cubin 的 PTX 段找 `cluster_dim` 指示。

## cluster 塌缩翻译器的伪代码

```python
def collapse_cluster_dims(ptx_input, target_arch="sm_120"):
    out = []
    cluster_was_active = False
    cluster_size = (1, 1, 1)

    for line in ptx_input:
        if line.startswith(".cluster_dim"):
            cluster_size = parse_cluster_dim(line)
            if cluster_size != (1, 1, 1):
                cluster_was_active = True
                out.append(".cluster_dim 1, 1, 1")
            else:
                out.append(line)

        elif "cluster_block_id" in line:
            if cluster_was_active:
                # 这个块原本想知道自己在 cluster 里的位置。
                # 既然塌缩了，答案永远是 0。
                out.append("    mov.b32 %ret, 0;    // collapsed cluster")
            else:
                out.append(line)

        elif "shared::cluster" in line:
            # 分布式共享内存访问——必须用全局内存模拟
            out.extend(emit_global_emulation(line))

        elif "tcgen05.mma.cta_group::2" in line:
            # 协作 MMA——必须拆开（无法机械翻译）
            out.append(f"// FATAL: cta_group::2 has no SM120 equivalent")
            out.append(f"// Original: {line}")
            raise NotMechanicallyTranslatable(line)

        else:
            out.append(line)

    return out
```

机械翻译对塌缩（思路 1）和全局内存模拟（思路 3）行得通。对协作计算（思路 2），自动翻译不现实：kernel 必须在源码层面拆开。

## cluster 是必需品而非便利品的情况

少数 kernel 是真的**需要** cluster 才能保证正确性，而不只是为了性能。例如：

- 某些 FlashAttention v3 变体，每个注意力 tile 横跨 2 个 CTA，才装得下 key/value
- 基于 cluster 共享 SMEM 的 ring-attention 实现
- 把 cluster barrier 当同步原语用的持久化 kernel

这些没有任何自动改写可行。kernel 需要人工重新设计一个单 CTA 版本。

## 另见

- [`blackwell/thread-block-clusters`](../blackwell/thread-block-clusters.md) — cluster 是什么
- [`translating-tcgen05`](translating-tcgen05.md) — `cta_group::2` MMA 对应的配套方案
- [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md) — cooperative groups 的背景
