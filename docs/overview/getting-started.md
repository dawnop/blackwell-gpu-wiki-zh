# 入门

本 wiki 本身的前置知识和阅读路线。这是一页"元"信息——告诉你读懂每一节之前需要先知道什么。

## 阅读前置知识

| 如果你已经了解…… | 就可以跳过…… |
| --- | --- |
| GPU 编程基础：warp、block、SM、共享内存 | [`fundamentals/`](../fundamentals/index.md) 的大部分内容，但建议浏览一下内存层级那页，看看 Blackwell 专属的数字 |
| CUDA C++ + PTX：`nvcc` 怎么产出二进制、`ptxas` 是干什么的 | [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md) |
| 到 Hopper 为止的 Tensor Core 编程（`wgmma.async`、TMA） | [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md)，但请仔细读 [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md)——那里面全是新东西 |
| 数值格式（FP16、BF16、FP8 E4M3/E5M2、FP4） | [`fundamentals/number-formats`](../fundamentals/number-formats.md) |
| 生产规模的 MoE 推理（DeepEP、NVSHMEM、EP/TP/PP） | [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) |

如果以上都不适用，就按侧边栏顺序从头读到尾。侧边栏是特意编排过的：每一页只引入下一页要用到的东西。

## 不需要搭建什么

整份 wiki 不碰 GPU 也能读完。所有概念都通过 PTX 清单、ISA 表格和对硬件的推理来说明——不运行任何代码，不启动任何 kernel。

这是有意为之。本 wiki 教的是问题的**形状**（ISA 不匹配、SMEM 上限、原子操作与总线拓扑）——而不是怎么操作某个具体工具。工具每两个月就换一茬，问题的形状不会变。

## 如果想在阅读之外动手，需要什么

如果你想在真实硬件上**验证**本 wiki 的论断，需要下面两者之一：

- **一块 SM 10.0 GPU**：B100、B200 或 B300（数据中心版 Blackwell）。云服务商（Lambda、RunPod、Together、AWS p5e 等）都有。用于案例分析和内核库的深入研究。
- **一块 SM 12.0 GPU**：RTX PRO 6000 Workstation、RTX 5090 或 RTX 5080。可以直接购买，也可以在一些工作站云服务商那里租。用于验证本 wiki 说会失败的东西，确实会失败。

两块不必都有，但把两块并排放着对比，是真正吃透它们之间差异的唯一办法。本 wiki 里大部分具体数字和观察结论都来自两者的对比。

工具方面：在任一张卡上装一个较新的 CUDA 工具包（12.8+，`sm_100a` / `sm_120a` 从 12.8 起才有），就能复现 PTX 生成和检查的示例。具体会用到：

- `nvcc`：编译
- `ptxas`：PTX 到 cubin 这一步
- `cuobjdump`：检查编译后的二进制
- `nvdisasm`：SASS 反汇编
- `nvidia-smi`：拓扑和总线层面的信息
- `nsys` 和 `ncu`（Nsight Systems/Compute）：性能归因

这些都不是读懂 wiki 的必需品。只是当你想离开页面、在自己的终端里验证点什么时，可以拿来用的选配工具。

## 推荐学习路线

根据目标不同，三条路线：

### "我想从头读到尾"

按侧边栏顺序读。总耗时：认真初读约 3–4 小时，重读约 2 小时。每一节末尾都有一个"检查点"问题——答得上来就往下走；答不上来就回头再读。

### "我有个具体的模型跑不起来"

1. 读 [`overview/index`](index.md) 和 [`overview/architecture`](architecture.md)（这两页就在你手边）。
2. 在 [`case-studies/`](../case-studies/index.md) 里找到你的模型（DeepSeek-V3/V4、Kimi-K2、GLM-5、通用）。
3. 案例分析会列出该模型假设的每一项具体特性。逐项点进链接，看解释它为什么在消费级 Blackwell 上不能用的那页。
4. 读 [`compatibility/`](../compatibility/index.md)，看替代方案长什么样。

耗时：专注阅读约 1 小时，顺着链接深入另计。

### "我是 CUDA/kernel 开发者，要在 SM100 和 SM120 之间移植代码"

1. [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md)——两者的详细差异。
2. [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md)——数据中心专属的新指令族。
3. [`kernels/cutlass`](../kernels/cutlass.md)——你最常需要移植的库。
4. [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md)——把 `tcgen05` 改写成 `mma.sync` 的套路。
5. [`compatibility/smem-budget-management`](../compatibility/smem-budget-management.md)——把 SMEM 用量压进 99 KiB 上限的套路。

耗时：约 2 小时，加上 wiki 指引你去读的源码。

## 检查点

当你不用回头翻就能回答下面这些问题时，这份 wiki 就算发挥作用了：

- `tcgen05.mma` 为什么在工作站 Blackwell 卡上不能用？
- 工作站 Blackwell 每 block 的共享内存上限是多少？数据中心版 Blackwell 又是多少？
- 一套在 NVL72 上跑到 49 tok/s 的 MoE 推理方案，为什么换到 4 卡 PCIe 拓扑上会跌到 1.4 tok/s？
- 要在消费级 Blackwell 机器上打开 PCIe 原子操作，需要哪两项设置？
- `sm_120f` 是什么意思？它和 `sm_120a` 有什么区别？

如果这些对你还是谜，那这份 wiki 还有活要干。

## 去哪里提问

本 wiki 是静态参考资料，没有讨论区。如果你发现错误或有疑问：

- NVIDIA 相关问题：NVIDIA 开发者论坛、CUTLASS / FlashInfer 的 GitHub issue 列表
- ML 系统相关问题：对应推理引擎项目的 issue 列表（sglang、vLLM、TRT-LLM）
- 更宽泛的架构问题：Frontier Forge、GPU MODE，或 ML 系统相关的 Discord 社区

[`reference/bibliography`](../reference/bibliography.md) 里的参考文献列出了本 wiki 引用的全部一手来源；对任何具体论断，先去查那些来源，不要默认 wiki 本身就是权威。
