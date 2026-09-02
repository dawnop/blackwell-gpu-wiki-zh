# EP 转 TP 的部署方案改写

收益最大的一条兼容方案：把专家并行（EP）的部署方案改成张量并行（TP）。

## 为什么重要

[`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) 里讲过，专家并行每层都要做一次 token 的 all-to-all。在没有 NVLink、也没有 P2P 原子操作的系统上，这次 all-to-all 得绕道主机桥，性能一落千丈。

张量并行每层只需要一次 all-reduce。NCCL 的环形 all-reduce 走 PCIe 和主机桥还算过得去，尤其是设了 NCCL_P2P_LEVEL=PIX 之后。

结果：把 EP 方案改写成 TP 方案，常常能砍掉 80% 以上的通信开销，而且一行 kernel 代码都不用动。

## 什么时候能改

三个条件：

1. **总显存够用。** TP 要把每个专家都切成 N 份分到所有 GPU 上。对一个有 E 个专家、每个专家权重为 `W_e` 的 MoE 模型，总权重是 `E × W_e`。TP 要求它能摊到 `N` 张 GPU 上，每张 `(E × W_e) / N`。

2. **模型不是围绕 EP 设计的。** 有些模型（比如带 NSA 的 DeepSeek-V4）的路由和特定的专家分布绑得很紧，改起来就难。

3. **推理引擎支持这种配置。** vLLM、sglang、TRT-LLM 都支持 MoE 模型只用 TP 的配置，但参数名和具体语义各不相同。

## 机械改写

概念上：

| EP 方案 | TP 方案 |
| --- | --- |
| 每张 GPU 持有互不重叠的一部分专家 | 每张 GPU 持有每个专家的一个 TP 分片 |
| 按 token 路由 → all-to-all | 按层 all-reduce（不做 token 路由） |
| 按 token 计的带宽：高 | 按层计的带宽：较低 |
| 显存：每张 GPU 占用少（只有部分专家） | 显存：每张 GPU 占用多（所有专家，分片存放） |

在推理引擎的配置里：

```yaml
# 改前（EP）
tensor_parallel_size: 1
expert_parallel_size: 4
moe_routing: standard
moe_all_to_all_backend: deepep   # 或 pplx，或基于 nvshmem 的后端

# 改后（TP）
tensor_parallel_size: 4
expert_parallel_size: 1            # 或者不写；默认为 1
disable_expert_parallelism: true
```

对于专家权重能沿隐藏维度干净切开的模型（MLP 的 up_proj、gate_proj、down_proj），这样直接换通常就能用。如果模型的共享专家跨越了专家边界，可能还需要引擎额外支持。

## 显存账

假设一个 MoE 模型：

- L 层
- 每层 E 个专家
- 每个专家权重 `W_e`（NVFP4 下每参数约 0.5 字节）
- 外加共享（非专家）权重 `W_shared`

**EP=N：**

```
每 GPU 权重 = (W_shared) + (E/N) × W_e × L
```

**TP=N：**

```
每 GPU 权重 = (W_shared / N) + E × (W_e / N) × L
           = (W_shared + E × W_e × L) / N
           = 总权重 / N
```

对大多数模型来说，**TP=N 每张 GPU 用的显存反而*更少***，因为共享权重也分片了。把共享权重要在每张卡上复制一份这笔账算进去，EP 的显存优势就是假象。

## 带宽账

每 token、每层：

**EP**：每个 token 的隐藏状态（H 字节）从当前 GPU → 专家所在 GPU → 再回来。每层两次 all-to-all。每 token 每层总字节数：约 2 × H ×（覆盖 top-k 个专家所需的专家跳数）。

**TP**：每层对激活 `(B × T × H)` 做一次 all-reduce。批大小 B = 1、序列长度 T = 1（decode）时，每层是 `H` 字节，但**分成小块沿环分摊**。有效的每层通信量：经过最慢链路的字节数为 `H × 2(N-1)/N`。

按典型数字（H = 8192，N = 4，top-k = 8）算：

- EP：约 16 × 8 × 8192 ≈ 1 MB 每层跨 GPU 流量
- TP：约 12 KB 每层跨 GPU 流量

差了 **约 80 倍**。这就是 TP 在消费级 Blackwell 上快得多的原因。

## 对吞吐的影响

一个在 B200 + NVLink 上用最优 EP 能跑到 100 tok/s 的模型，在工作站 Blackwell（PCIe + 主机桥）上用 EP 可能只有 5 tok/s。同一个模型在工作站 Blackwell 上改用 TP，常常能到 50–70 tok/s——找回约 10 倍。

## EP 转 TP 的几个坑

- **路由 kernel 的变化。** EP 会启动一个路由 kernel，为每个 token 挑专家。TP 用不着——每张 GPU 都有全部专家，路由是本地的。有些引擎把路由 kernel 写死了，要确认引擎在 TP 模式下真的跳过了它。

- **激活显存。** TP 有时会抬高激活的峰值显存（每张 GPU 先算出完整的隐藏激活，再为下一层分片）。小心 OOM。

- **数值精度。** TP 的 all-reduce 只是近似满足结合律；EP 是精确的（没有跨 GPU 归约）。对某些模型，TP 会引入细微的数值差异。要测输出是否等价，不能只看有没有 NaN。

- **微批。** EP 鼓励大的 per-token 批（摊薄 all-to-all 的开销），TP 不需要。改完之后可能要重新调微批大小。

## 一个方案改写器的伪代码

```python
def rewrite_ep_to_tp(model_config, num_gpus):
    """
    把一份使用 EP 的模型配置改写成纯 TP 部署。
    """
    if model_config.parallelism.ep_size <= 1:
        return model_config    # 已经是纯 TP

    new = copy.deepcopy(model_config)
    new.parallelism.tp_size = num_gpus
    new.parallelism.ep_size = 1
    new.parallelism.disable_expert_parallelism = True

    # 有些引擎的路由路径有单独的开关
    new.engine.moe_routing_kernel = "local"   # 而不是 "all_to_all"
    new.engine.moe_all_to_all_backend = None

    # 检查显存预算
    total_weight = compute_total_weight(model_config)
    per_gpu_after_tp = total_weight / num_gpus
    if per_gpu_after_tp > GPU_MEMORY * 0.94:
        warn("TP may not fit; consider reducing context or using PP")

    return new
```

## 纯 TP 装不下怎么办

如果模型太大，按你的 GPU 数量做完整 TP 也装不下，可以退到：

- **TP × PP 混合。** 把层切到成对的 GPU 上（PP），每对内部用 TP。
- **剪枝版本。** REAP-160 那样的剪枝能去掉约 1/3 的专家，质量损失很小。
- **更低精度。** 混合精度（部分层用 Marlin 跑 W4A16，其余用 NVFP4）可以省显存。

这些是体面的降级，不算失败。

## 另见

- [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) — 为什么 EP 这么吃带宽
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md) — 为什么 all-to-all 在工作站 Blackwell 上这么难
- [`kernels/inference-engines`](../kernels/inference-engines.md) — 各引擎做这项改写的具体开关
- [`case-studies/glm-5`](../case-studies/glm-5.md) — 一个实际应用 EP 转 TP 的例子
