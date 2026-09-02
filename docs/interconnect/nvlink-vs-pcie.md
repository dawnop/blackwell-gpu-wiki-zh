# NVLink 与 PCIe

GPU 之间通信用的两种互连。NVLink 快但是私有的；PCIe 通用但慢（一些）。工作站 Blackwell 只有 PCIe。

## NVLink —— 高端方案

NVLink 是 NVIDIA 私有的 GPU 到 GPU 互连。第 5 代（Blackwell）是当前这一代。

| 代际 | 单链路带宽 | 单 GPU 带宽 | 架构 |
| --- | ---: | ---: | --- |
| NVLink 1 | 20 GB/s | 80 GB/s | Pascal（P100） |
| NVLink 2 | 25 GB/s | 300 GB/s | Volta（V100） |
| NVLink 3 | 50 GB/s | 600 GB/s | Ampere（A100） |
| NVLink 4 | 100 GB/s | 900 GB/s | Hopper（H100） |
| **NVLink 5** | **200 GB/s** | **1.8 TB/s** | **数据中心版 Blackwell（B100/B200）** |

推理关心的是单 GPU 带宽：即一张 GPU 同时与所有对端交换数据的总速率。B100 的 1.8 TB/s 足以完全喂饱 200k token 注意力对 KV 带宽的需求，还绰绰有余。

NVLink 端点是 GPU 板卡上的物理接口。两种用法：

- **NVLink Bridge**：用一个无源桥接器直连 2 张 GPU。工作站上用（RTX A6000 及更早几代）；RTX PRO 6000 工作站 Blackwell **没有**这个选项。
- **NVSwitch**：一颗交换芯片，在单个机箱内把 8 张以上 GPU 互相连起来。DGX/HGX 机器用的就是它。

一种特殊形态是 **MNNVL**（Multi-Node NVLink，多节点 NVLink），NVL72 系统（NVIDIA 的机架级平台）上有。MNNVL 把 NVLink fabric 扩展到多个机箱，最多让 72 张 GPU 呈现为一个 NVLink 域。

## NVSwitch 与拓扑变体

在一台 DGX H100（8× H100）里：

```
  H100  H100  H100  H100  H100  H100  H100  H100
   |     |     |     |     |     |     |     |
   +---NVSwitch----+----NVSwitch--------+
        (共 4 颗 NVSwitch，网状互连)
```

每张 GPU 经由交换芯片到任何其他 GPU 都有完整的 NVLink 带宽。带宽是均匀的（没有远近之分）。这是大多数现代 MoE 算法所假设的"理想"拓扑。

NVL72 通过 MNNVL 把这个规模扩到机架级的 72 张 GPU。

## PCIe —— 通用的回退方案

PCIe 是标准的主机侧互连。GPU 通过 PCIe 连到 CPU；在多 GPU 系统里，GPU 之间也可以借 PCIe 直接对话（peer-to-peer，即 P2P）。

| 代际 | 单通道带宽 | x16 单向 | x16 双向 |
| --- | ---: | ---: | ---: |
| PCIe 3.0 | 1.0 GB/s | 16 GB/s | 32 GB/s |
| PCIe 4.0 | 2.0 GB/s | 32 GB/s | 64 GB/s |
| PCIe 5.0 | 4.0 GB/s | 64 GB/s | 128 GB/s |
| PCIe 6.0 | 8.0 GB/s | 128 GB/s | 256 GB/s |

一条 Gen4 x16 插槽的单向带宽是 **32 GB/s**，对比 NVLink 5 的单 GPU 带宽 **1.8 TB/s**，大约**慢 55 倍**。

PCIe Gen5 把这个数翻倍，但消费级工作站主板插满多卡时，出于 PCB / 信号完整性的限制，经常会降速到 Gen4。所以一张 RTX 5090 插在 4 槽主板上，即使两端都支持 Gen5，实际也可能跑在 PCIe 4.0 x16。

## 拓扑限制

PCIe 系统有**根复合体（root complex）**——CPU 通常带多个 PCIe 根复合体，各自驱动一部分插槽。同一根复合体内的 GPU 可以直接做 P2P 传输；跨根复合体的传输可能要**经主机内存中转**（更慢）。

一台典型的 4 GPU 工作站：

```
+-[0000:00] Root Complex 0     ← GPU0
+-[0000:40] BMC root           ← (没有 GPU)
+-[0000:80] Root Complex 2     ← GPU1, GPU2
+-[0000:c0] Root Complex 3     ← GPU3, GPU4
```

P2P 矩阵：

- GPU1 ↔ GPU2：PHB（PCIe Host Bridge，PCIe 主机桥）—— **可以走快速 P2P**
- GPU3 ↔ GPU4：PHB —— **可以走快速 P2P**
- GPU0 ↔ 其他任何 GPU：NODE（AMD 上要过一跳 Infinity Fabric，Intel 上是 QPI）—— **更慢，有时需要主机中转**
- GPU1 ↔ GPU3（跨根复合体）：NODE —— **同上**

可以用下面的命令查看：

```bash
nvidia-smi topo -m
```

在一台工作站 Blackwell 机器上：

```
        GPU0  GPU1  GPU2  GPU3
GPU0    X     PHB   NODE  NODE
GPU1    PHB   X     NODE  NODE
GPU2    NODE  NODE  X     PHB
GPU3    NODE  NODE  PHB   X
```

对 NCCL 来说，这意味着跨根复合体的传输走的是更慢的路径。环境变量 `NCCL_P2P_LEVEL` 控制哪些 GPU 对会去尝试 P2P：

- `NCCL_P2P_LEVEL=PIX` —— 只限同一个 PCIe 内部交换芯片下
- `NCCL_P2P_LEVEL=PXB` —— 同一根复合体（PCIe 桥级别）
- `NCCL_P2P_LEVEL=NODE` —— 同一 NUMA 节点（允许跨根复合体，可能要中转）
- `NCCL_P2P_LEVEL=SYS` —— 整个系统范围

工作站 Blackwell 上一个常见的坑：NCCL 默认会尝试跨根复合体做 P2P，结果死锁，推理引擎在预热阶段超时。设置 `NCCL_P2P_LEVEL=PIX` 可以避开，它强制跨根复合体的传输走主机中转。

## ReBAR —— 可调整大小的 BAR

一项 PCIe 特性，允许主机把整块 GPU 显存映射进 BAR（Base Address Register，基地址寄存器）空间，这样主机（以及其他 GPU）就能直接寻址 GPU 的全部显存。没有 ReBAR 时，只暴露一个很小的窗口（256 MiB）。

MoE 通过 P2P 做 all-to-all 时，ReBAR 是**必需的**——kernel 需要往对端 GPU 显存里写，而目标地址超出了默认小窗口的范围。

开启 ReBAR 的步骤：

1. **BIOS**：在主板设置里打开 "Above 4G Decoding" 和 "Resizable BAR"
2. **GPU 固件**：必须支持 ReBAR（大多数 Blackwell 卡都支持）
3. **驱动**：NVIDIA 驱动 470+ 支持 ReBAR

开启 ReBAR 的一个副作用：PCIe **总线号可能会变**，因为 BAR1 区域变大了很多，BIOS 会重新分配总线号。任何硬编码了总线 ID（比如 `01:00.0`）的脚本，开启 ReBAR 后可能需要改。

## 什么时候 PCIe "够用"

PCIe Gen4 / Gen5 对下面这些完全够用：

- **张量并行**：每层的 all_reduce 数据量很小
- **流水线并行**：相邻阶段之间的 P2P 发送
- **单 GPU 推理**：根本没有 GPU 间流量

对下面这些勉强够用：

- **小 batch 的 MoE**：每步的 all-to-all 数据量小，但对延迟敏感
- **长上下文 TP decode**：与 KV 相关的集合通信比较敏感

对下面这些就不行了：

- **大 batch 的 EP MoE**：all-to-all 的数据量把 PCIe 打满，吞吐崩掉
- **任何依赖 P2P 原子操作的东西**：见 [`p2p-and-atomics`](p2p-and-atomics.md)

## 需要记住的几个数字

- **NVL72 NVLink 5**：1.8 TB/s/GPU
- **DGX 级 NVSwitch**：900 GB/s/GPU
- **PCIe Gen5 x16**：64 GB/s 单向
- **PCIe Gen4 x16**：32 GB/s 单向
- **跨根复合体、经主机中转**：实际 8–16 GB/s
- **NVLink 5 与 PCIe Gen4 的带宽比**：约 55 倍

最后这个数就是 [`moe-parallelism`](moe-parallelism.md) 里 EP 相对 TP 崩掉的那个数量级。

## 另见

- [`p2p-and-atomics`](p2p-and-atomics.md) —— P2P、原子操作与 ACS
- [`moe-parallelism`](moe-parallelism.md) —— 面对带宽差距该怎么办
- [`kernels/nvshmem-and-deepep`](../kernels/nvshmem-and-deepep.md) —— 面向 NVLink 级 fabric 的那些库
- *NVIDIA NVLink and NVSwitch documentation*
- *PCIe Specification 4.0 / 5.0*
