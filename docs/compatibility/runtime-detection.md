# 运行时检测

系统启动时怎样探测 GPU 和拓扑，自动分发到正确的代码路径。

## 要检测什么

一次完整的探测要回答这些问题：

| 问题 | API |
| --- | --- |
| 每张 GPU 是什么架构？ | `cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMajor/Minor)` → SM120 返回 `(12, 0)` |
| 每张 GPU 有多少显存？ | `cudaMemGetInfo` |
| 每个 SM 有多少 SMEM 可用？ | `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` |
| 最大 cluster 是多大？ | `cudaDeviceGetAttribute(cudaDevAttrClusterLaunch)` + 试探性探测 |
| 有没有 TMEM？ | 间接判断：仅 SM100 / SM103（B200 / B300）有 |
| NVLink 接上了吗？ | `nvmlDeviceGetNvLinkState`、`nvidia-smi nvlink --status` |
| GPU A 能直接访问 GPU B 的显存吗？ | `cudaDeviceCanAccessPeer(A, B)` |
| GPU A 能对 GPU B 的显存做原子操作吗？ | **没有直接的 API**。做一次原子操作再检查结果。 |
| PCIe 拓扑是什么样？ | `nvidia-smi topo -m` |

最后一项——跨 P2P 的原子操作——最棘手，因为没有干净的 API。务实的做法：启动一个小 kernel，对远端显存做 `atomicAdd`，检查它是硬件完成的，还是掉到了主机模拟。如果结果正确而且很快，原子操作可用；如果报错或者挂住，就不可用。

## 探测器的伪代码

```python
class GPUProbe:
  def detect(self):
    info = {}
    n = cuda.device_count()
    info["device_count"] = n
    info["devices"] = []

    for i in range(n):
      d = {}
      d["arch_major"], d["arch_minor"] = cuda.compute_capability(i)
      d["arch_string"] = f"sm_{d['arch_major']}{d['arch_minor']}"
      d["memory_total"] = cuda.mem_info(i)[1]
      d["smem_per_block_optin"] = cuda.attr(i, "MaxSharedMemoryPerBlockOptin")
      d["sm_count"] = cuda.attr(i, "MultiProcessorCount")
      d["max_cluster_size"] = self._probe_cluster_size(i)
      d["has_tmem"] = d["arch_string"] in {"sm_100", "sm_103"}  # B200 / B300
      info["devices"].append(d)

    info["nvlink_links"] = self._probe_nvlink()
    info["p2p_matrix"] = self._probe_p2p(n)
    info["p2p_atomics"] = self._probe_p2p_atomics(n)
    info["pcie_topology"] = self._probe_pcie_topology()

    return info

  def _probe_cluster_size(self, dev):
    # 试着以 cluster_dim 2 启动一个 kernel，看能不能成功
    for size in [16, 8, 4, 2, 1]:
      if try_launch_with_cluster_dim(dev, size):
        return size
    return 1

  def _probe_p2p(self, n):
    m = [[False] * n for _ in range(n)]
    for i in range(n):
      for j in range(n):
        if i != j:
          m[i][j] = cuda.can_access_peer(i, j)
    return m

  def _probe_p2p_atomics(self, n):
    # 从 GPU i 启动一个 kernel，对 GPU j 的显存做 atomicAdd，
    # 校验结果，并给这次操作计时。
    m = [[None] * n for _ in range(n)]
    for i in range(n):
      for j in range(n):
        if i != j:
          result = try_p2p_atomic_add(src=i, dst=j)
          m[i][j] = result  # 可能是 "hardware"、"host_fallback" 或 "fail"
    return m
```

## 启动时怎么用探测结果

典型的推理引擎这样消费探测结果：

```python
probe = GPUProbe().detect()

# 按架构分发
arch = probe["devices"][0]["arch_string"]
if arch == "sm_120":
  cutlass_template_tree = "sm120_optimized"
  use_deepgemm = False
  use_tcgen05 = False
elif arch == "sm_100":
  cutlass_template_tree = "sm100_optimized"
  use_deepgemm = True
  use_tcgen05 = True

# 按 SMEM 预算选模板
smem_per_kernel = probe["devices"][0]["smem_per_block_optin"]
if smem_per_kernel < 102400:  # 小于 100 KiB
  select_smaller_tile_templates()

# 按拓扑定并行方案
if not probe["nvlink_links"]:
  # 没有 NVLink：避开 EP，优先 TP
  parallelism_plan.disable_ep = True
elif probe["p2p_atomics"][0][1] != "hardware":
  # 没有 P2P 原子操作：避开 one-shot all-to-all
  flashinfer_one_shot_a2a = False
```

探测在启动时只跑一次；结果缓存下来，用于配置后续所有 kernel 启动。

## 最难探测的一项：P2P 原子操作

跨 P2P 的原子操作是对 FlashInfer 的 MoE one-shot all-to-all 影响最大的能力（见 [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md)），偏偏它没有直接的 API。

务实的探测方法：

```cuda
__global__ void probe_atomic_kernel(int *dst, int *flag, int expected) {
  atomicAdd(dst, 1);
  __threadfence_system();
  *flag = 1;
}

bool probe_p2p_atomics_works(int src_dev, int dst_dev) {
  // 在 dst_dev 上分配 dst
  cudaSetDevice(dst_dev);
  int *d_counter, *d_flag;
  cudaMalloc(&d_counter, sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMemset(d_counter, 0, sizeof(int));

  // 从 src_dev 启动
  cudaSetDevice(src_dev);
  auto start = clock_now();
  probe_atomic_kernel<<<1024, 256>>>(d_counter, d_flag, 0);
  cudaDeviceSynchronize();
  auto elapsed = clock_now() - start;

  // 把结果读回来
  int counter_val;
  cudaMemcpy(&counter_val, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

  if (counter_val == 1024 * 256) {
    // 所有原子操作都成功了
    return elapsed < THRESHOLD_HARDWARE;
    // 如果 elapsed > 阈值，原子操作多半走了主机模拟
  }
  return false;  // 有原子操作丢了；不安全
}
```

这个阈值用来区分"硬件原子操作"（约 µs 级）和"回退到主机的原子操作"（约 ms 级）。在工作站 Blackwell 上，不在同一个 PCIe 交换机下的两张 GPU 之间，这个探测通常返回"主机回退"或"失败"。

## 缓存与稳定性

正常情况下探测结果在运行时不会变，所以要缓存起来：

- 存到文件（`/tmp/gpu_probe.json`，以主机名 + GPU UUID 作键）
- 在进程生命周期内保留

以下情况重新探测：

- GPU 变了（驱动重载、热插拔——少见）
- 内核模块版本变了
- 用户明确要求重新探测

## 报告

一个好用的探测器还会生成一份人能读的报告：

```
GPU Probe Report
================
Hostname: workstation-1
Detected 4 GPUs:

 GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation
     arch: sm_120, memory: 96 GB, SMEM/block: 99 KiB
 GPU 1: same
 GPU 2: same
 GPU 3: same

NVLink: not detected on any pair
P2P matrix:
    0  1  2  3
  0  -  Y  Y  Y   (all PCIe Gen4)
  1  Y  -  Y  Y
  2  Y  Y  -  Y
  3  Y  Y  Y  -

P2P atomics: HOST FALLBACK on all pairs (avg latency 4.2 ms)

Recommendations:
 - Use TP-only parallelism (avoid EP)
 - Disable FlashInfer one-shot all-to-all
 - Disable DeepGEMM
 - Set NCCL_P2P_LEVEL=PIX
 - Use SM120-targeted CUTLASS templates
 - Use Triton attention with kv_splits=64
```

启动探测应该把这份报告交给用户（或写进日志），让他们明白自动配置是怎么定下来的。

## 另见

- [`compatibility/ep-to-tp-rewriting`](ep-to-tp-rewriting.md) — 检测到没有 NVLink 之后该做什么
- [`compatibility/smem-budget-management`](smem-budget-management.md) — 检测到 SM120 之后该做什么
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md) — 原子操作为什么重要的硬件原因
- [`kernels/inference-engines`](../kernels/inference-engines.md) — 各引擎怎么消费探测结果
