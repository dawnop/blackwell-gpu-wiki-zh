# 案例分析

具体的前沿模型为什么在工作站 Blackwell 上跑不起来，以及每个模型能让我们看清架构分裂的哪一面。

## 本节页面

- [`deepseek-v3-v4`](deepseek-v3-v4.md) —— 最典型的"EP 跑在 NVLink 上"的设计
- [`kimi-k2`](kimi-k2.md) —— Moonshot 的 MoE 系列，情况类似
- [`glm-5`](glm-5.md) —— 智谱的 MoE 系列，对 `tcgen05` 的依赖没那么重
- [`generic-moe-on-consumer-blackwell`](generic-moe-on-consumer-blackwell.md) —— 总结：一份适用于任何新 MoE 模型的检查清单

## 为什么挑这几个模型

这几个模型都是 2024 年中到 2026 年初之间由前沿实验室发布的，部署平台都瞄准数据中心版 Blackwell。它们代表了现代开放权重 MoE 模型的典型样子：

- 总参数 100B–700B，每 token 激活 30B–60B
- 64–256 个专家，top-k=2 到 top-k=8 路由
- 发布时就附带 NVFP4 量化版本
- 参考部署用 DeepGEMM、FlashInfer、sglang 或 vLLM
- 性能数据在 B100 或 H200 上测得

共同的模式是：**参考部署默认你用的是数据中心版 Blackwell**。放到工作站 Blackwell 上，每次暴露出来的都是同一批问题，只是各模型有些细节差异。

## 怎么用这一节

每篇案例分析回答四个问题：

1. **这是个什么模型？**（架构、规模、设计用途）
2. **它的参考部署默认了什么？**（硬件、kernel、并行方式）
3. **在工作站 Blackwell 上哪里会坏？**（具体哪个 kernel、具体怎么坏）
4. **怎么绕过去？**（配置、kernel 替换、并行方案改写）

如果你要部署某个具体模型，在这里找到它，读对应的页面。如果你的模型不在列表里，[`generic-moe-on-consumer-blackwell`](generic-moe-on-consumer-blackwell.md) 给出了一套诊断流程，可以自己照着做。

## 一点整体观察

所有案例最后都收敛到同一个答案：**只用 TP 并行 + NVFP4（或 W4A16）权重 + FP8 KV + 高 kv-splits 的 Triton attention + 关掉 DeepGEMM。**

到 2026 年中，任何工作站 Blackwell 上的 MoE 部署基本都会走到这套配方。它不是最优的——和数据中心部署相比吞吐差 30–50 倍——但它是**能跑起来**的那套配置，而这正是大多数用户需要的。

部署的其余细节（具体 tile 形状、具体 FP8 KV 布局、具体 page size）因模型而异，但大方向都在往这个模式收敛。
