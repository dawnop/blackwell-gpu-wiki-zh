# 参考资料

查阅用的材料——缩写表、参考文献，以及推荐的阅读顺序。

## 页面

- [`abbreviations`](abbreviations.md) —— 全站用到的缩写和简称速查
- [`bibliography`](bibliography.md) —— 论文、NVIDIA 文档、kernel 库源码和值得一读的博客的引用

术语表（带完整的文字定义）见 [`overview/glossary`](../overview/glossary.md)。

## 推荐阅读顺序

### "我想弄明白为什么我的 MoE 模型在工作站版 Blackwell 上这么慢"

1. [`overview/architecture`](../overview/architecture.md) —— 核心论点
2. [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md) —— 硬件差异
3. [`interconnect/nvlink-vs-pcie`](../interconnect/nvlink-vs-pcie.md) —— 带宽差异
4. [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) —— MoE 打算怎么用这些带宽
5. [`compatibility/ep-to-tp-rewriting`](../compatibility/ep-to-tp-rewriting.md) —— 解决办法

### "我想从头到尾理解这块 GPU 本身"

1. [`fundamentals/gpu-execution-model`](../fundamentals/gpu-execution-model.md)
2. [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md)
3. [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md)
4. [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md)
5. [`fundamentals/number-formats`](../fundamentals/number-formats.md)
6. [`blackwell/`](../blackwell/index.md)（整个章节）

### "我想了解各个 kernel 库"

1. [`fundamentals/`](../fundamentals/index.md)（整个章节）
2. [`kernels/`](../kernels/index.md)（整个章节，按顺序读）
3. [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) —— 需要移植某个库时看
4. [`compatibility/smem-budget-management`](../compatibility/smem-budget-management.md) —— 约束条件那一面

### "我只想要个太长不看版"

[`overview/architecture`](../overview/architecture.md)。读那一页就行，5 分钟概括整个 wiki。
