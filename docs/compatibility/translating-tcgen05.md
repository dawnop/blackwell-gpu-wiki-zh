# 把 tcgen05 翻译成 mma.sync

把只能在数据中心版 Blackwell 上跑的 PTX（用 `tcgen05` 指令族）改写成工作站版 Blackwell 也能跑的 PTX（用 `mma.sync` 指令链）的套路。

## 形状映射

一条 `tcgen05.mma` 指令干的活，相当于很多条 `mma.sync`。翻译的本质是**形状拆分**：

| `tcgen05.mma` 形状 | 等价的 `mma.sync` 条数 | `mma.sync` 形状 |
| --- | --- | --- |
| `m64n64k16`（FP16） | 16 | m16n8k16（m、n 方向 4×4 网格；k 方向 1 份） |
| `m64n64k32`（FP8） | 8 | m16n8k32（m、n 方向 4×4；k 方向 0.5× 缩放） |
| `m64n64k64`（FP4） | 4 | m16n8k32（m、n 方向 4×4；k 方向分两趟累加） |
| `m128n128k64`（FP4，单 CTA） | 16（n 方向再 ×2） | m16n8k32 |
| `m128n256k64`（FP4，单 CTA） | 32 | m16n8k32 |
| `m256n128k64`（FP4，**CTA pair**） | （SM120 上没有单 CTA 等价物） | —— |

最大的单 CTA `tcgen05.mma`（m128n256k64）拆开后，每个累加器 tile 对应 32 条 `mma.sync m16n8k32`。配上流水线还能接受；不配流水线就全部串行。

最大的 `tcgen05.mma.cta_group::2` 形状（m256n128k64）**没有单 CTA 等价物**。要翻译它，你必须：

- 把工作切成两半
- 每一半按单 CTA tile 处理
- 再把结果拼起来

这比单纯的形状拆分侵入性大得多。

## 一个完整的翻译示例

原始 SM100 PTX：

```ptx
// 为累加器分配 16 KB TMEM
.reg .b32 %tmem_d_addr;
tcgen05.alloc.cta_group::1 %tmem_d_addr, 16384;

// 发射 MMA：D = A * B + D，NVFP4 输入，FP32 累加器
tcgen05.mma.cta_group::1.kind::nvf4
    [%tmem_d_addr],         // 累加器位置（TMEM）
    [%smem_a_desc],         // A 描述符（SMEM）
    [%smem_b_desc],         // B 描述符（SMEM）
    %scale_a,
    %scale_b;

// 等待完成
.reg .b64 %sema;
tcgen05.commit.cta_group::1 %sema;
tcgen05.wait.cta_group::1 %sema;

// 把结果从 TMEM 拷到 SMEM，交给下游
tcgen05.cp.tmem.shared::cta.b64 [%smem_out], [%tmem_d_addr];

// 释放 TMEM
tcgen05.dealloc %tmem_d_addr, 16384;
tcgen05.relinquish_alloc_permit;
```

翻译后的 SM120 PTX（示意）：

```ptx
// 在 SMEM 里分配同样大小的空间（算在 99 KiB 预算里）
.shared .align 16 .b32 smem_d_buf[4096];   // 16 KB / 每个 FP32 4 字节

// 把 A、B 操作数从 SMEM 加载到寄存器
.reg .b32 %ra<8>;
.reg .b32 %rb<8>;
.reg .f32 %rd<32>;     // 累加器放在寄存器里（分摊到各线程）

// 初始化累加器（或从上一轮累加器加载）
mov.f32 %rd0, 0.0;
// ... %rd1 到 %rd31 同理 ...

// 发射一串 mma.sync m16n8k32（NVFP4 → FP32）
mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32
    {%rd0, %rd1, %rd2, %rd3},      // 累加器输出
    {%ra0, %ra1},                   // 操作数 A（NVFP4 打包）
    {%rb0, %rb1},                   // 操作数 B（NVFP4 打包）
    {%rd0, %rd1, %rd2, %rd3};      // 累加器输入

// ... 其余 tile 还有 31 条类似的 mma.sync ...

// 应用缩放因子
.reg .f32 %scale_combined;
mul.f32 %scale_combined, %scale_a, %scale_b;
mul.f32 %rd0, %rd0, %scale_combined;
// ... 对 %rd1 到 %rd31 同样处理 ...

// 写 SMEM 前先同步 warp
bar.sync 0;

// 把累加器存到 SMEM（整个 warp 一起）
st.shared.f32 [%smem_d_buf+0],   %rd0;
st.shared.f32 [%smem_d_buf+128], %rd1;     // 每个线程存自己的 tile
// ... 依此类推
```

翻译后的 PTX **明显更长**：约 50 行，原来只有约 20 行。指令数也多得多（每线程 32 条 mma.sync，外加缩放逻辑）。

## 对性能的影响

`tcgen05.mma` 是异步的：warp 发射之后就继续往下走。而一串 `mma.sync` 是同步的：每一条都会阻塞 warp 直到执行完。

要在 SM120 上把重叠找回来，kernel 必须：

1. 在外层循环的多次迭代之间，把操作数加载和 MMA **做成流水线**
2. 在 MMA 指令链开始之前，**用软件预取**把操作数提前搬进 SMEM/寄存器
3. 把工作**分摊到同一个 CTA 的多个 warp**，每个 warp 跑自己的 MMA 指令链

CUTLASS 的 SM120 模板正是这么做的，这也是为什么它们和 SM100 的模板是两棵独立的模板树。两棵树的差别不是小修小补，而是两种不同的 kernel 设计。

直接硬翻（不做流水线）的 kernel，大概只能达到 SM120 最优吞吐的 **30–50 %**。仔细做好流水线之后能到 **60–75 %**。剩下那段差距纯粹是指令发射开销，没有 `tcgen05` 就物理上绕不过去。

## TMEM 换成 SMEM 或寄存器

TMEM 分配是翻译的第二个难点。有三种策略：

### 策略 1：寄存器

对于小累加器（m64 一级），FP32 累加器可以放进寄存器。一个 m64n64k32 tile 产生 64×64 = 4096 个 FP32 值 × 4 字节 = 16 KB。分摊到 128 线程的块上，每线程 128 字节 = 32 个 32 位寄存器，可行。

这是最干净的翻译：TMEM 分配直接变成寄存器声明，完全不占 SMEM。

### 策略 2：SMEM 暂存

对于大累加器（m128 一级），64 KB 寄存器放不下，就暂存到 SMEM：

```ptx
.shared .align 16 .b32 smem_accumulator[16384];  // 64 KB

// 内层循环里：
ld.shared.f32 %rd0, [smem_accumulator + offset];
// ... mma.sync 累加进 %rd0 ...
st.shared.f32 [smem_accumulator + offset], %rd0;
```

这会吃掉 99 KiB SMEM 预算中的 64 KB，只剩 35 KiB 给操作数暂存。非常紧，往往只能把流水线深度砍下来。

### 策略 3：分块

把 m128 tile 拆成 4 个 m64 tile，用寄存器累加器依次处理。用吞吐换 SMEM 预算的余量。

## 缩放因子的小坑

NVFP4 的缩放因子需要特殊处理。`tcgen05.mma.kind::nvf4` 指令把缩放因子寄存器当作输入，在 Tensor Core 内部就把它乘上去了。而 `mma.sync.m16n8k32.f32.e2m1.e2m1.f32` 只接收原始 FP4 输入，不带内置缩放。

翻译时把缩放放到 MMA 之后做一次乘法：

```ptx
// mma.sync 指令链结束后：
mul.f32 %rd_out, %rd_acc, %scale_a_combined;
mul.f32 %rd_out, %rd_out, %scale_b_combined;
```

或者提前乘：在 MMA 之前先把操作数缩放好，代价是损失精度。

## Cluster pair 的 MMA：没有干净的翻译

`tcgen05.mma.cta_group::2` 在两个协作的 CTA 上发射一个 m256 一级的 MMA。没有哪条单 CTA 的 `mma.sync` 能在一次启动里产出同样的 tile。

现实可行的翻译：

1. 把 m256 tile 拆成两个 m128 tile
2. 作为独立的 CTA 分别启动（不需要 cluster，它们互不依赖）
3. 在更高层面把两者的输出合并（例如都写到全局内存，让下一层读合并后的结果）

这比其他套路侵入性都大：它改的不只是 PTX，而是 kernel 的启动结构。

## 通用翻译器的伪代码

```python
def translate_tcgen05(ptx_input, target_arch="sm_120"):
    instructions = parse_ptx(ptx_input)
    output = []
    tmem_to_smem = {}    # TMEM 地址到 SMEM 分配的映射

    for instr in instructions:
        if instr.op == "tcgen05.alloc":
            smem_alloc = allocate_smem(instr.size)
            tmem_to_smem[instr.dst_reg] = smem_alloc
            output.append(decl_smem(smem_alloc))

        elif instr.op == "tcgen05.mma":
            shape = instr.tile_shape
            mma_chain = decompose_to_mma_sync(shape, instr.kind)
            output.extend(mma_chain)

        elif instr.op == "tcgen05.cp.tmem.shared":
            smem_src = tmem_to_smem[instr.src_addr]
            output.append(make_shared_to_shared_copy(smem_src, instr.dst_smem))

        elif instr.op in ("tcgen05.commit", "tcgen05.wait"):
            # mma.sync 指令链是同步的；除了 bar.sync 不需要别的屏障
            output.append("bar.sync 0;")

        elif instr.op == "tcgen05.dealloc":
            pass    # SMEM 分配随作用域自动释放

        elif instr.op == "cluster_dim 2,1,1":
            output.append("cluster_dim 1,1,1")
            # 警告：如果 kernel 依赖 cluster 协作，必须把它拆开

        else:
            output.append(instr)    # 非 tcgen05 指令原样透传

    return emit_ptx(output)
```

这只是概念示意。真实实现（不管是 CUTLASS 里的、Triton 编译器里的，还是独立工具）要处理多得多的情况：流水线用的 mbarrier、异步 TMA、缩放因子格式转换、寄存器压力分析等等。

## 自动翻译搞不定的情况

以下几种情况没法机械地翻译：

- **kernel 用了 `tcgen05.shift`**（TMEM 布局变换，SMEM 里没有对应物）
- **kernel 依赖 `cta_group::2` 协作**（没有单 CTA 翻译）
- **kernel 用了 cluster 共享的 TMA**（`cp.async.bulk.tensor.shared::cluster.global`，需要拆 cluster）

遇到这些，唯一的办法是在源码层面手工重写。

## 另见

- [`smem-budget-management`](smem-budget-management.md) —— 翻译中 SMEM 预算这一侧的问题
- [`cluster-rewriting`](cluster-rewriting.md) —— `cta_group::2` 的翻译
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) —— `tcgen05` 是什么
- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) —— `mma.sync` 的背景知识
- *NVIDIA PTX ISA 8.5*，Tensor Core 指令一节
