# SMEM 预算管理

怎样把 kernel 塞进 SM120 更紧的共享内存（SMEM）预算里。

## 数字

| 架构 | 每 SM 总 SMEM | 驱动预留 | kernel 可用 | 主要差别 |
| --- | --- | --- | --- | --- |
| SM90（H100） | 228 KiB | 约 1 KiB | **227 KiB** | 基线 |
| SM100（B100/B200） | 228 KiB | 约 3 KiB | **225 KiB** | 微乎其微 |
| SM120（RTX PRO 6000） | 100 KiB | 约 1 KiB | **99 KiB** | **−128 KiB** |

一个在 SM100 上能跑、但用了 100–225 KiB SMEM 的 kernel，不缩减 SMEM 占用就**没法**在 SM120 上启动。

## SMEM 都花在哪

一个典型的 MoE GEMM kernel 里：

```
kernel 的 SMEM 总用量 =
    操作数 A 暂存缓冲区
  + 操作数 B 暂存缓冲区
  + 累加器暂存空间（如果不在寄存器 / TMEM 里）
  + 缩放因子缓冲区（NVFP4：每 16 个元素 1 字节）
  + barrier / mbarrier
  + 每 warp 的暂存空间（如果有）
  + epilogue 暂存空间（例如做激活函数融合用）
```

以一个 4 级流水线的 m128n256k64 NVFP4 GEMM 为例：

| 组成部分 | SM100 上的大小 | SM120 上的大小 |
| --- | --- | --- |
| 操作数 A：128 × 64 × 0.5 B × 4 级 | 16 KiB | （相同） |
| 操作数 B：64 × 256 × 0.5 B × 4 级 | 32 KiB | （相同） |
| A 和 B 的缩放因子：约 4 KiB | 4 KiB | （相同） |
| 累加器（FP32）：128 × 256 × 4 B | （在 TMEM 里，0 KiB） | **128 KiB** |
| barrier / mbarrier | 1 KiB | （相同） |
| **SMEM 总计** | 53 KiB | **181 KiB → 超出 99 KiB** |

累加器就是那道断崖。在 SM100 上它住在 TMEM 里，不占 SMEM；到了 SM120 上，它只能住在 SMEM 或寄存器里。

## 塞进去的策略

### A. 缩小 tile 形状

把 tile 在某一个维度上减半：

| 形状变化 | 累加器缩减 | 吞吐代价 |
| --- | --- | --- |
| m128n256 → m128n128 | 128 → 64 KiB | 每个 CTA 的吞吐约减半，但 CTA 数翻倍来覆盖同样的工作 |
| m128n256 → m64n256 | 128 → 64 KiB | 类似 |
| m128n256 → m64n128 | 128 → 32 KiB | 每个 CTA 只剩四分之一，但 CTA 数变成 4 倍 |

m64n128 的 tile 能留出充足的 SMEM 余量，而且因为 SM 占用率更高（同时驻留的 CTA 更多），性能往往和 m128n256 相差无几。

### B. 把累加器挪到寄存器

一个 m64n64 的 FP32 累加器是 16 KB → 4096 个 32 位寄存器，分摊到 128 个线程上 = 每线程 32 个寄存器。紧，但可行。

```cuda
// 每线程的寄存器累加器
float acc[8][4];   // 每个线程持有 8×4 = 32 个 FP32 值
```

tile 再大，寄存器压力就成了新的断崖：每线程超过约 96 个寄存器后，占用率会急剧下降。

### C. 减少流水线级数

4 级流水线要保存 4 份操作数暂存缓冲区。减到 2 级，操作数占的 SMEM 就减半。

| 流水线级数 | 操作数 SMEM | 延迟隐藏效果 |
| --- | --- | --- |
| 4 级 | 48 KiB（按我们的例子） | 极好 |
| 3 级 | 36 KiB | 很好 |
| 2 级 | 24 KiB | 还行 |
| 1 级 | 12 KiB | 没有，内存延迟完全暴露 |

3 级通常是个不错的折中：延迟隐藏效果大部分保留，SMEM 省下 25 %。

### D. 重算而不是暂存

对某些 kernel（例如 FlashAttention），中间值可以重新算一遍，而不是暂存起来。这是用算力换 SMEM。SM120 的算力相对 SMEM 的比例更高，这笔买卖往往划算。

### E. 溢出到 L2 而不是 SMEM

对于外层循环多次迭代之间反复用到的值，可以写到全局内存（会落在 L2 里），而不是暂存在 SMEM。L2 很大（工作站版 Blackwell 是 96 MB，B200 更多），可以当成一块更慢但更大的 SMEM 来用。延迟大约 150 个周期，而 SMEM 只有约 30 个周期，所以只适合复用间隔足够长的值。

## 预算工作表

对每个 kernel 算一遍预算：

```
预算 = 99 KiB
  - 1 KiB 驱动预留
  - 1 KiB barrier/mbarrier
  = 97 KiB 可用于数据
```

然后分门别类：

```
操作数 A：        [size] × [pipeline_depth]
操作数 B：        [size] × [pipeline_depth]
缩放因子：        [scale_size] × [pipeline_depth]
累加器：          [acc_size]   （如果不在寄存器/TMEM 里）
epilogue 暂存：   [epilogue_size]
                  ─────────────────────
总计：            [sum]    必须 ≤ 97 KiB
```

如果总计 > 97 KiB，就套用策略 A–E，然后重新算。

## 感知 SMEM 的 kernel 选择器伪代码

```python
def select_kernel_for_sm120(M, N, K, dtype):
    """
    针对给定的 GEMM 形状和数据类型，挑一个能塞进
    99 KiB SMEM 预算的 CUTLASS 风格 kernel 模板。
    """
    candidates = enumerate_sm120_templates(M, N, K, dtype)
    feasible = []
    for tmpl in candidates:
        smem_use = compute_smem_use(tmpl)
        if smem_use <= 97 * 1024:
            feasible.append((tmpl, smem_use, estimate_throughput(tmpl)))

    if not feasible:
        # 回退到更小的 tile
        return fallback_small_tile_template(M, N, K, dtype)

    # 挑估计吞吐最高的那个
    return max(feasible, key=lambda x: x[2])
```

## 常见情形："能塞下，但就差一点"

很多 SM100 kernel 翻译到 SM120 后，落在 95–105 KiB 这个区间，*刚好*超一点预算。这种情况下，典型的修法是砍掉一级流水线（比如 4 级减到 3 级），省下约 12 KiB。这比缩小 tile 形状温和得多。

如果这样还塞不下，下一步是把 tile 的某一个维度减半。如果*这样*还塞不下，kernel 就得大改设计了。

## 验证

套用完这些策略之后：

1. 用 `nvcc --gpu-architecture=sm_120 --ptxas-options=-v` **编译** kernel，从 ptxas 的输出里读 SMEM 数字。
2. **确认**它 ≤ 99 KiB。如果 ptxas 报的数字 > 99 KiB，kernel 启动时会报"out of resources"错误。
3. **做性能分析**，确认缩小后的 tile 仍能达到可接受的吞吐。通常做到 SM100 吞吐的 60–75 % 是现实的。

## 另见

- [`translating-tcgen05`](translating-tcgen05.md) —— 配套的套路；tcgen05 翻译往往会增加 SMEM 用量
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) —— 为什么 TMEM 对 SMEM 预算很重要
- [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md) —— 完整的内存层次图景
- [`kernels/cutlass`](../kernels/cutlass.md) —— CUTLASS 如何把这些取舍暴露给用户
