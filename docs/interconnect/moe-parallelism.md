# MoE 并行

混合专家（Mixture-of-Experts）模型是怎么切分到多张 GPU 上的，以及为什么在工作站 Blackwell 上，切分方案的选择决定了吞吐。

## 几种方案速览

| 方案 | 切的是什么 | 每层通信什么 | 带宽需求 |
| --- | --- | --- | --- |
| **TP（张量并行）** | 每个权重矩阵切到 N 张 GPU 上 | 注意力输出和 FFN 输出之后各一次 `all_reduce` | 低到中等 |
| **PP（流水线并行）** | 层切成 N 个阶段，微批次逐段流过 | 相邻阶段之间 P2P send/recv | 低 |
| **EP（专家并行）** | 每个 MoE 层的专家切到 N 张 GPU 上 | token 的 dispatch 和 combine 各一次 `all_to_all` | **高** |
| **DP（数据并行）** | 模型复制，切 batch | 每层一次 `all_reduce`（训练）；推理中少见 | 低 |

稠密模型（Llama、Mistral）只有 TP 和 PP 适用。MoE 模型（DeepSeek、Mixtral、Kimi、GLM-5）多了 EP 这个选项，可以和 TP、PP 组合使用。

## EP 是怎么工作的

一个 MoE 层有 N 个专家，每个专家是一个前馈网络。路由把每个 token 分配给 N 个专家中的 top-k 个（一般 k=2 或 8）。

EP=N 时，专家 *i* 放在 GPU *i* 上。对每个 token：

1. **路由**：一次很小的计算（一个 softmax），各 GPU 共享
2. **Dispatch**：把 token 发给它的 top-k 个专家（跨 GPU）
3. **计算**：每个专家对分到自己的 token 跑 FFN
4. **Combine**：把专家输出送回 token 原来所在的 GPU（跨 GPU）

第 2 步和第 4 步是 **all-to-all** 操作，也是主要开销。

## MoE 里 TP 是怎么工作的

TP=N（不用 EP）时，每个专家的权重矩阵切到全部 N 张 GPU 上。**每张 GPU 都持有每个专家的一片。**路由把 token 分配给专家后，每张 GPU 算自己那一片 FFN，最后一次 `all_reduce` 把各片加起来。

步骤：

1. **路由**：同上
2. **Dispatch**：在每张 GPU 内部对 token 做一次置换（dispatch 本身没有跨 GPU 流量）
3. **计算**：每张 GPU 算每个专家 FFN 的自己那一片，再乘上路由权重
4. **归约**：N 张 GPU 之间做一次 `all_reduce`（每层一次）

通信是 `all_reduce`（数据量 = hidden_dim × tokens），而不是 `all_to_all`（数据量 = N × hidden_dim × tokens）。搬的数据大约**少 N 倍**。

## 为什么带宽差距这么要紧

一个典型的 MoE 推理步：

| 量 | 值 |
| --- | --- |
| 隐层维度 | 7168 |
| 每步 token 数 | 64 |
| GPU 数 | 4 |
| 专家数 | 256 |
| Top-k | 8 |

EP 每层的 all-to-all 数据量：光 dispatch 就是 `4 × 7168 × 64 × 2 字节（BF16）≈ 3.7 MB`。combine 再翻一倍。大约**每层 8 MB**。

TP 每层的 all_reduce 数据量：`7168 × 64 × 2 字节 ≈ 920 KB`。大约**每层 1 MB**。

每步通信量之比：EP 大约多 8 倍。

但那只是数据量。更大的问题是**带宽利用**：

- NVLink 5：约 1.8 TB/s。8 MB 在 **约 5 µs** 内传完。EP 没问题。
- PCIe Gen4：约 32 GB/s。8 MB 要 **约 250 µs**。EP 成了最大的开销。

对一个 100 层的模型，这意味着 PCIe 上每个 token 有 25 ms 纯通信时间——无论算得多快，decode 都超不过约 40 tok/s。换成 TP，同样的模型每 token 只要个位数毫秒。

## 实测的崩塌

在一台 4 GPU 的工作站 Blackwell 上跑一个 478B 参数的 MoE 模型，实测：

| 方案 | Decode tok/s |
| --- | ---: |
| TP=4（无 EP） | ~49 |
| EP=4（经 NCCL 在 PCIe 上做 all-to-all） | ~1.4 |

一个配置项的差别，**慢了 35 倍**。整个"让这个模型在工作站 Blackwell 上跑起来"的工程，归结起来就是：把推理引擎配成 TP=4，彻底不用 EP。

## 什么时候绕不开 EP

有些 MoE 模型是为 EP 设计的，没有干净的 TP 回退方案。两种常见情形：

### "共享专家"加"路由专家"

DeepSeek-V2/V3 把专家分成：
- **共享专家**：小，每个 token 都用（适合 TP）
- **路由专家**：大，稀疏激活（天然适合 EP）

对这类模型，路由专家有 NVLink 时最好用 EP，没有就用 TP。

### N 特别大（专家数量很多）

N > 64 个专家的模型可能塞不进纯 TP 方案，因为每张 GPU 都要复制每个专家的权重——N 很大时内存根本装不下。

一个 512 专家、每个专家约 5 GB 的模型，TP 下每张 GPU 要放 2.5 TB 的专家权重。只能用 EP。

对能装进工作站 Blackwell 的 MoE 模型（一般是 100B–700B、64–256 个专家）来说，得益于 NVFP4 量化省下的大量内存，纯 TP 方案**是**放得下的。

## 混合方案

典型的生产部署既不是纯 TP 也不是纯 EP，而是混合：

- **TP × EP**：专家切到一部分 GPU 上（EP），每个专家的权重再切到其余 GPU 上（TP）。大型数据中心部署里很常见。
- **TP × PP**：阶段内张量并行，阶段间流水线并行。超大稠密模型的标准做法。
- **TP × EP × PP**：三者都上。万亿参数 MoE 的标准做法。

对没有 NVLink 的工作站 Blackwell：

- **纯 TP** 是默认选择
- **TP × PP** 用于内存吃紧时（靠 PP 分段能多装些模型，代价是延迟略增）
- **不用 EP**，除非你已经解决了原子操作的问题（就算解决了，上限也是 NCCL 回退那条路）

## 看懂推理引擎的并行参数

sglang 里：

```bash
--tensor-parallel-size 4         # TP=4
--pipeline-parallel-size 1       # PP=1
# （没有 EP 的参数；sglang 根据模型和拓扑自己决定）
```

vLLM 里：

```bash
--tensor-parallel-size 4
--pipeline-parallel-size 1
--enable-expert-parallel false   # 显式关闭 EP（vLLM 0.7+）
```

"EP 开关"有时隐含在模型配置里。DeepSeek-V3 的参考配置默认用 EP，你可能需要显式覆盖掉。

## 判断当前生效的是哪种方案

看启动日志：

```bash
grep -iE 'tensor.parallel|pipeline.parallel|expert.parallel' /tmp/sglang.log
```

或者跑一次带埋点的前向，看 NCCL 流量：

```bash
NCCL_DEBUG=INFO bash launcher.sh 2>&1 | grep -i 'all_to_all\|all_reduce\|nccl'
```

如果每层的热路径里出现 `all_to_all` 调用，说明 EP 在生效。如果只有 `all_reduce`，就是纯 TP。

## 对内存的影响

各方案的内存占用不同：

| 方案 | 每 GPU 模型内存 |
| --- | --- |
| TP=N | total_model / N（模型被切开） |
| EP=N（无 TP） | total_model / N（专家被切开） |
| TP=N × EP=M | total_model / (N × M) |
| 复制专家（每 GPU 内做 TP） | total_model / N，**但每张 GPU 上都有全部专家** |

在"复制专家"模式下（每张 GPU 有全部专家的权重，但每个专家的权重是 TP 切开的），每 GPU 内存是 `(non_expert_weights / N) + (sum_of_all_experts / N)`。总量一样，只是切法不同。

对专家占大头的 MoE 模型（比如权重内存的 90 %），TP 复制专家方案和 EP 方案的每 GPU 内存大致相同。区别在通信方式：TP 的 all_reduce 对 EP 的 all-to-all。

## 小结

工作站 Blackwell 上 MoE 并行方案的选择：

- **默认**：TP，不用 EP，不用 PP
- **内存吃紧**：加 PP（延迟略增，可用内存更多）
- **真的需要 EP**：只在原子操作已开启的前提下用，并接受性能被 PCIe 封顶
- **绝不**：基于 NVSHMEM 的 EP（DeepEP 节点内模式），除非你有 NVLink

在这类硬件上，最大的一根性能杠杆就是"别用 EP"。

## 另见

- [`nvlink-vs-pcie`](nvlink-vs-pcie.md) —— 这个选择背后的带宽数字
- [`p2p-and-atomics`](p2p-and-atomics.md) —— 是什么挡住了依赖原子操作 kernel 的 EP
- [`kernels/nvshmem-and-deepep`](../kernels/nvshmem-and-deepep.md) —— 会跑不起来的那些库
- [`compatibility/ep-to-tp-rewriting`](../compatibility/ep-to-tp-rewriting.md) —— 改写的套路
- *DeepSeek-V3 Technical Report*（最初在 NVL72 上做 EP 的设计）
