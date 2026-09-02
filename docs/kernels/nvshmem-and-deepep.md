# NVSHMEM 与 DeepEP

大规模 MoE 用的 GPU 到 GPU 通信原语。两者的设计都默认有 NVLink 级别的互连 fabric；在工作站 Blackwell 上要么性能很差，要么干脆跑不起来。

## NVSHMEM

NVIDIA 在 GPU 上实现的 OpenSHMEM 标准：在一个分区全局地址空间（PGAS）上做**单边**的 GPU 内存操作（put、get、原子操作）。

GitHub：不公开；随 HPC SDK 发布。协议：NVIDIA 专有。

### 它做什么

典型的 NVSHMEM 调用：

```c
nvshmem_putmem(dst_ptr_on_peer, src_ptr_local, size, peer_pe);
nvshmem_signal_op(flag_ptr_on_peer, value, op, peer_pe);
nvshmem_signal_wait_until(flag_ptr_local, op, value);
```

这三行：

1. 发起一次从本地 GPU 内存到对端 GPU 内存的异步单边写
2. 在对端置一个标志（一次远程原子更新）
3. 等待本地标志被置位（对原子变量忙等轮询）

关键在于：从发起方 GPU 的角度看，这个操作是**发出去就不管了**——发起方立刻继续往下走。同步靠的是 signal/wait 这一对。

### 为什么 MoE all-to-all 要用它

在带专家并行（EP）的 MoE 层里，每个 token 都要被路由到它分配到的专家那里，而这些专家（可能）在别的 GPU 上。dispatch 就是一次 all-to-all：

```
对每个 rank r：
    对每个对端 p：
        把（路由到 p 的 token）发给对端 p
```

朴素实现是一串 NCCL `send`/`recv` 调用，但那是*双边*的（发送方和接收方都要发起操作）。NVSHMEM 的单边模型更高效：发送方发出全部写操作，接收方只需轮询完成标志。稳态下这能**把通信开销减半**。

DeepEP——DeepSeek 的专家并行 all-to-all kernel——正是因为这个原因才使用 NVSHMEM。

### SM 兼容性

原则上 NVSHMEM 在任何 CUDA 架构上都能用。但性能随互连不同差别巨大：

| 拓扑 | NVSHMEM 吞吐 |
| --- | --- |
| NVLink 5（NVL72） | 每 GPU 约 1.8 TB/s，延迟约 1 µs |
| NVSwitch（DGX） | 每 GPU 约 900 GB/s |
| PCIe Gen5 P2P（数据中心） | 每对 GPU 约 64 GB/s |
| **PCIe Gen4 P2P（消费级）** | **每对 GPU 约 32 GB/s，延迟高得多** |

PCIe 路径是存在的，但比 NVLink 慢一个数量级。更糟的是，NVSHMEM 的 signal-wait 依赖 **P2P 原子操作**，而消费级显卡在软件层面把它关掉了（见 [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md)）。

### 对工作站 Blackwell 意味着什么

即使 NVSHMEM 在技术上可用，工作站 Blackwell 上的实际体验也是：

- **没有 NVLink → 带宽只有 PCIe 水平（慢约 50 倍）**
- **没有 P2P 原子操作 → signal/wait 死锁**

所以基于 NVSHMEM 的 MoE all-to-all 在工作站 Blackwell 上实际上是不可用的。变通办法：改用 NCCL all-to-all，或者换并行方案，干脆绕开 all-to-all。

## DeepEP

DeepSeek 的专家并行 all-to-all kernel 套件。用 NVSHMEM（节点内）和 RDMA（节点间）实现 MoE EP 层的 dispatch 和 combine 两个阶段。

GitHub：`deepseek-ai/DeepEP`。协议：MIT。

### 它提供什么

三种传输方式：

- **节点内（intranode）**：NVSHMEM 走 NVLink，单机箱内。快速路径，面向 DGX 级硬件。
- **节点间（internode）**：RDMA 走 InfiniBand 或 RoCE，多节点。扩展路径，面向配有 RDMA 网卡的数据中心集群。
- **Hybrid-EP**（实验性）：PCIe + NVSHMEM 混合。成熟度较低；设计意图是能在偏消费级的拓扑上工作。

### SM 兼容性（以及拓扑兼容性）

| 配置 | DeepEP 节点内 | DeepEP 节点间 |
| --- | --- | --- |
| NVL72（NVLink + MNNVL） | ✓ 最优 | ✓ 最优 |
| DGX H100/H200 | ✓ | ✓ 通过 RDMA 网卡 |
| 8× H100 PCIe + RDMA | 部分可用（没有 NVLink，必须走节点间路径） | ✓ |
| **4× 工作站 Blackwell，无 NVLink，无 RDMA** | ✗ 需要 NVLink | ✗ 需要 RDMA 网卡 |

对工作站 Blackwell 这种情况，**DeepEP 的两种传输方式都用不了**。Hybrid-EP 是理论上唯一的选项；但它是实验性的，也没在消费级 Blackwell 上验证过。

### 常见故障

**在工作站 Blackwell 上**：

- DeepEP 节点内初始化失败，因为 NVSHMEM 检测不到 NVLink 端点
- DeepEP 节点间初始化失败，因为没有 RDMA 网卡
- Hybrid-EP 路径可能能启动，但输出错误或者挂死

推荐做法是在工作站 Blackwell 上**根本不要用 DeepEP**，把 EP 方案换成不需要 all-to-all 的 TP+PP 方案。

## 用 NCCL 回退

NCCL（NVIDIA Collective Communications Library）提供标准的集合通信操作：`all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all`。和 NVSHMEM 不同，NCCL 是**双边**的，而且能跑在任何后端上（NVLink、PCIe、IB，甚至 TCP）。

在工作站 Blackwell 上做 MoE all-to-all，NCCL 的 `all_to_all_single` 是实际可用的实现：

```python
torch.distributed.all_to_all_single(output, input, output_split_sizes, input_split_sizes)
```

它比 NVLink 上的 NVSHMEM 慢（没有单边操作的好处，也不是常驻 SM 执行），但不需要原子操作就能正确工作。

NCCL 在工作站 Blackwell 上也有自己的怪癖：

- 建议设 `NCCL_P2P_LEVEL=PIX`（只在同一个 PCIe 内部交换机下尝试 P2P；跨根复合体的传输经主机中转）
- 没有 IB 就设 `NCCL_IB_DISABLE=1`
- 注意 TP=4 预热时的跨根复合体死锁；有时设 `SGLANG_PYNCCL_SKIP_WARMUP=1`（或 vLLM 里的对应变量）能解决

## 汇总表

| 库 | 拓扑假设 | 工作站 Blackwell？ |
| --- | --- | --- |
| NVSHMEM | 靠 NVLink 拿性能 + 靠原子操作做同步 | 没有原子操作就坏掉；就算有，也非常慢 |
| DeepEP 节点内 | NVSHMEM + NVLink | 需要 NVLink |
| DeepEP 节点间 | RDMA 网卡 | 需要 RDMA |
| DeepEP hybrid-EP | PCIe（实验性） | 未经验证 |
| NCCL | 任意后端，双边 | 能正确工作，比 NVLink 上的 NVSHMEM 慢 |
| FlashInfer one-shot a2a | 经 P2P 的原子操作 | 没有原子操作就坏掉 |

**结论**：目前发布的每一个"快速"MoE all-to-all kernel 都假定有 NVLink 或 PCIe 原子操作二者之一。工作站 Blackwell（默认情况下）两者都没有。NCCL 回退能用，但 EP 相对 TP 的性能优势也就所剩无几了。

这就是为什么消费级 Blackwell 上的 MoE 推理最后都用 TP+PP 而不是 EP——见 [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md)。

## 另见

- [`interconnect/nvlink-vs-pcie`](../interconnect/nvlink-vs-pcie.md)——为什么光是带宽就很要紧
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md)——原子操作这道坎
- [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md)——EP 与 TP 的取舍
- *NVIDIA NVSHMEM Programming Guide*
- GitHub 上的 `deepseek-ai/DeepEP`
