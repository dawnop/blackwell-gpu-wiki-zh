# DeepGEMM

DeepSeek-AI 出品的高吞吐 FP8/FP4 GEMM 库，专为 DeepSeek-V2/V3/V4 推理中出现的那些 GEMM 形状而设计。**截至 2026 年初只支持 SM100**，SM120 移植正在进行中。

GitHub：`deepseek-ai/DeepGEMM`。协议：MIT。由 DeepSeek-AI 维护。

## 是什么

一个独立的 GEMM 库，思路上和 CUTLASS 类似，但更小、更专注，只为一个具体场景优化：**FP4/FP8 精度的 MoE 推理**，尤其看重分组 GEMM（grouped GEMM）的性能——也就是各"组"之间 N 或 M 维度不同的情况，例如 MoE 层里按专家分别做的 GEMM。

这个库在 JIT 阶段从 Python 模板生成 CUDA kernel，支持的 tile 形状和流水线配置只有一小组。

## 为什么不直接用 CUTLASS

CUTLASS 是通用的，DeepGEMM 是有的放矢的。具体来说：

- **分组 GEMM**：DeepGEMM 处理逐专家 GEMM 比 CUTLASS 的 `GroupedGemm` 模板更高效，因为它把 kernel 启动开销摊得更彻底
- **NVFP4 细节**：原生支持 DeepSeek 使用的 NVFP4 布局（他们预量化权重的格式）
- **tile 种类更少**：DeepGEMM 只提供 DeepSeek 模型会用到的那些形状，调优起来更简单
- **`tcgen05` 优先**：从一开始就是为 SM100 写的，以 tcgen05/TMEM 为中心的设计是刻在骨子里的

## 依赖什么

- CUDA toolkit（SM100 需要 ≥ 12.4）
- 一个支持 `--gpu-architecture=compute_100` 的 `nvcc`
- PyTorch（用于 Python 绑定）

## SM100 的情况

完整支持，且高度优化。DeepGEMM 在 B100 / B200 上能逼近 FP4 峰值吞吐。这些 kernel：

- 以 `sm_100a` 为编译目标
- 使用 `tcgen05.mma.cta_group::1` 和 `cta_group::2`
- 把累加器放在 TMEM 里
- 用 cluster 共享的 TMA 搬运操作数
- 首次使用时 JIT 编译，缓存在 `~/.cache/deepgemm/`

DeepGEMM 是 SM100 原生库的典型代表之一。读它的源码是学习现代数据中心版 Blackwell kernel 设计的好途径。

## SM120 的情况

**发布版本：不支持。** DeepGEMM 的 `gemm_jit.py` 默认是：

```python
gencode_flags = [
    "-gencode", "arch=compute_100,code=sm_100a",
]
```

在工作站 Blackwell 显卡上加载 DeepGEMM 会得到：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**移植正在进行中。** 这项工作并不轻松，因为每个 kernel 都直接使用 `tcgen05`：

```cuda
// DeepGEMM kernel 里这类内联 PTX 的示意（译注：原文的指令拼法不存在，已按 PTX ISA 改写）：
asm volatile(
    "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n"
    : : "r"(smem_slot)                       // TMEM 地址会被写到这个 SMEM 位置
);
asm volatile(
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n"
    : : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(tmem_sfa), "r"(tmem_sfb)
);
```

移植需要：

1. 把每条 `tcgen05.mma` 换成一串 `mma.sync` 指令
2. 把 TMEM 分配换成寄存器或 SMEM 分配
3. 缩小 tile 形状，塞进 SM120 的 99 KiB SMEM 上限
4. 避开 `cta_group::2`（SM120 没有 CTA pair MMA；cluster 本身可以留）

这**不是重新编译一下就完事**，而是要大幅重写 kernel 内层循环。所以移植才这么费时间。

## 眼下怎么办

对于运行那些依赖 DeepGEMM 的模型的工作站 Blackwell 用户：

**方案 1**：换成 CUTLASS 的 NVFP4 GEMM。CUTLASS 的 SM120 NVFP4 模板输出是正确的（SMEM 断崖的问题除外）。吞吐比 SM100 上跑 DeepGEMM 低，但和 DeepGEMM 移植到 SM120 之后能达到的水平相比差不了多少。

大多数使用 DeepGEMM 的推理引擎都有 CUTLASS 回退路径：

```bash
# 以 sglang 为例：
SGLANG_DISABLE_DEEP_GEMM=1
SGLANG_ENABLE_DEEP_GEMM=0
```

这些环境变量关掉 DeepGEMM 的 dispatch，把 MoE GEMM 转到 CUTLASS 上。

**方案 2**：最重的那几个 GEMM 换成 Marlin（INT4）。精度更低、占用内存带宽更少，在 SM120 上工作正常。

**方案 3**：用 FP8 而不是 FP4 做服务。权重更大（内存约 2 倍），但完全不依赖 NVFP4 路径。CUTLASS 里有 SM120 的 FP8 GEMM kernel，也没有断崖问题。

## 常见故障

**故障 1：`no kernel image`**——DeepGEMM 的 cubin 只有 SM100 版本。见上文。

**故障 2：缩放因子布局不匹配**——DeepGEMM 使用的是某种特定形式的 NVFP4 布局（块交错，FP8 E4M3 缩放因子）。如果模型文件是按 MX-FP4 布局（OCP 标准，块大小 32，E8M0 缩放因子）保存的，通过 DeepGEMM 加载会悄无声息地算出垃圾。（译注：原文把 MX-FP4 的缩放因子写成 FP6 E3M2，按 OCP 规范已改为 E8M0。）有些模型两种布局都提供，要选对。

**故障 3：半途而废的移植尝试污染了 JIT 缓存**

如果你试过打补丁让 DeepGEMM 以 `sm_120` 为目标，JIT 缓存里可能留着编译了一半的垃圾。清掉它：`rm -rf ~/.cache/deepgemm/`。

## 检测方法

```bash
python -c "import deep_gemm; print(deep_gemm.__file__, deep_gemm.__version__)"
ls ~/.cache/deepgemm/
```

如果缓存里只有 `100a/` 子目录，没有 `120/` 或 `120a/`，说明你用的是未移植的版本。

## 阅读 DeepGEMM 源码

```
deep_gemm/
├── csrc/                       # C++ kernel 源码
├── deep_gemm/jit/              # Python JIT 框架（gemm_jit.py 等）
├── tests/                      # 逐 kernel 的正确性测试
└── tools/                      # 基准测试
```

先看 `deep_gemm/jit/gemm_jit.py`，了解架构目标的选择逻辑；再看 `csrc/` 里真正的 kernel。

## 更深层的影响

DeepGEMM 是"模型发布依赖一个只提供 `sm_100a` 的 kernel 库"的典型例子。DeepSeek V3 和 V4 的发布说明都把 DeepGEMM 列为推荐的 GEMM 后端；这个推荐在 B100/B200 上没问题，在 RTX PRO 6000 Workstation 上却会悄悄失效。

[`case-studies/`](../case-studies/index.md) 里一半的案例分析背后都是这个套路：前沿实验室的模型加上它的参考部署栈，默认你用的是数据中心版 Blackwell。kernel 库生态（首当其冲是 DeepGEMM，但 FlashInfer-MoE 和 DeepEP 也一样）还没跟上消费级 Blackwell，尽管两者的硬件在很多方面很相似。

## 另见

- [`cutlass`](cutlass.md)——SM120 上的替代 GEMM 后端
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md)——DeepGEMM 所依赖的东西
- [`blackwell/nvfp4-deep-dive`](../blackwell/nvfp4-deep-dive.md)——格式本身
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md)——移植套路
- GitHub 上的 `deepseek-ai/DeepGEMM`
- DeepSeek-V3 和 DeepSeek-V4 的模型发布博客
