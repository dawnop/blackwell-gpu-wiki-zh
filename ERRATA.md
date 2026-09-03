# 相对原始出处的修正与新增

本仓库的中英文正文都以这里记录的状态为准。"原文"指 `main` 分支保存的原始英文快照。
核对依据：NVIDIA PTX ISA（9.3）、CUDA C++ Programming Guide、Blackwell Compatibility / Tuning Guide、
OCP Microscaling (MX) 规范、NVIDIA 公开产品规格、CUTLASS 文档，以及 `~/workspace/logo/b200-features.md` 的调研。

## 第一轮（2026-09-02）：贯穿多页的三类错误

### SM120 支持 cluster

原文全站的核心前提之一是"工作站版 Blackwell（SM 12.0）不支持线程块簇，cluster 只能为 1"，并由此推出
"声明 cluster 的 kernel 在 SM120 上会被静默降级、死锁或算错"。这与 NVIDIA 文档不符：

- CUDA C++ Programming Guide 的 "Feature Support per Compute Capability" 表里，Thread Block Cluster 和
  Distributed Shared Memory 两行在 9.0、10.x、12.x 都是 Yes。
- Blackwell Tuning Guide："Thread block clusters are supported by Blackwell GPUs as well"；可移植上限 8，
  非可移植 16 只有 B200 可以开。
- NVIDIA 开发者论坛有人在 RTX 5090 上以 4 个 block 的 cluster 跑通 `cluster.map_shared_rank`。

SM120 相对 SM100 真正缺的是建在 cluster 之上的两样东西：`tcgen05.mma.cta_group::2`（因为没有 tcgen05）和
硬件加速的 multicast TMA（PTX ISA 注明 `.multicast::cluster` 只针对 sm_90a / sm_100a 系列优化）。

改动页面：`blackwell/thread-block-clusters`、`blackwell/sm100-vs-sm120`、`compatibility/cluster-rewriting`、
`fundamentals/gpu-execution-model`、`kernels/cutlass`、`blackwell/tcgen05-and-tmem`、`blackwell/index`、`kernels/deepgemm`。

### `wgmma.async` 不在 Blackwell 上

原文多处说 `wgmma.async` 能在 SM 10.0 和 SM 12.0 上跑。实际上 `wgmma` 是 `sm_90a` 专属：
数据中心版 Blackwell 换成了 `tcgen05.mma`，工作站版 Blackwell 只有 `mma.sync`（含 `sm_120a` 专属的块缩放版）。

改动页面：`fundamentals/tensor-cores`、`blackwell/sm100-vs-sm120`、`blackwell/tcgen05-and-tmem`、`blackwell/index`、
`overview/glossary`、`kernels/cutlass`、`kernels/flashattention`、`kernels/triton-and-transformerengine`。

### `tcgen05` 指令的拼法和语义

原文的指令表和所有 PTX 示例用了不存在的拼法：按字节分配 TMEM、`tcgen05.wait`、TMEM→SMEM 方向的 `tcgen05.cp`、
`kind::nvf4` / `kind::f4`、把缩放因子当寄存器操作数等。按 PTX ISA 改写：

- `tcgen05.alloc` 按**列**分配（32–512 之间的 2 的幂），TMEM 地址写进 SMEM；整个 warp 一起执行。
- 没有 `tcgen05.wait`：`tcgen05.commit` 把已发射的异步操作挂到 **mbarrier** 上。
- `tcgen05.cp` 只有 SMEM→TMEM 一个方向；TMEM 和寄存器之间用 `tcgen05.ld` / `tcgen05.st`。
- MMA 类型是 `kind::f16 / tf32 / f8f6f4 / i8 / mxf8f6f4 / mxf4 / mxf4nvf4`；NVFP4 走 `kind::mxf4nvf4.block_scale`，缩放因子放在 TMEM。
- `tcgen05.mma` 由**单个线程**发射，A、B 通过 SMEM 矩阵描述符给出。
- TMEM 是 128 lane × 512 列 × 32 位；原文的"默认/步长/复制/压缩"四种布局和"`tcgen05.shift` 做布局变换"不存在。
- CTA pair：两个 CTA 各出一半 SMEM 操作数、各收一半 TMEM 结果，由 leader CTA 的一个线程发射一次。
- tile 上限：单 CTA M=128、N=256；CTA pair M=256、N=256（原文写 128×128 / 256×128）。
- 引入版本：PTX ISA 8.6（随 CUDA 12.8），原文写 8.4/8.5。

改动页面：`blackwell/tcgen05-and-tmem`、`fundamentals/tensor-cores`、`fundamentals/memory-hierarchy`、
`fundamentals/cuda-pipeline`、`compatibility/translating-tcgen05`、`kernels/deepgemm`、`blackwell/nvfp4-deep-dive`、
`blackwell/sm100-vs-sm120`、`overview/glossary`。

## 第一轮：单点错误

| 页面 | 原文 | 改为 |
| --- | --- | --- |
| `overview/architecture` | 计算能力"小数点后是主版本号" | 小数点前是主版本号 |
| `overview/glossary`、`fundamentals/cuda-pipeline` | `sm_NNf` = "forward-compatible" | NVIDIA 的叫法是 family-specific |
| `overview/glossary` | PCIe Gen4 / Gen5 每 lane 16 / 32 GB/s | 每 lane 16 / 32 GT/s（约 2 / 4 GB/s 每方向），x16 约 32 / 64 GB/s 每方向 |
| `fundamentals/memory-hierarchy` | m256n128k64 累加器 32 KB | m128n256 为 128 KB，m256n256 为 256 KB |
| `fundamentals/memory-hierarchy` | Hopper/SM100 的 L1+SMEM 合计 228 KiB | 合计 256 KB，其中最多 228 KB 可配给 SMEM |
| `fundamentals/number-formats` | FP32 是"1962 年的 IEEE 格式" | IEEE 754 是 1985 年 |
| `fundamentals/number-formats`、`blackwell/nvfp4-deep-dive`、`kernels/deepgemm`、`kernels/cutlass`、`overview/glossary` | MX-FP4 缩放因子是 FP6 E3M2，每元素 4.19 位 | OCP MX 规范是 E8M0（8 位），每元素 4.25 位 |
| `fundamentals/tensor-cores` | 每个 Tensor Core 每周期发射一条 m128n128k64；B100 144 个 SM、约 5 PFLOPs | 按 B200 公开峰值反推约每 SM 每周期 1.6 万次 FP4 乘累加；B200 148 个 SM |
| `fundamentals/tensor-cores` | RTX PRO 6000 FP4 约 125 TFLOPs，与 B100 相差 40 倍 | RTX PRO 6000 FP4 4000 TOPS（稀疏）；B200 18/9 PFLOPS；差约 4–5 倍，折算到每 SM 每周期约 8 倍 |
| `blackwell/sm100-vs-sm120`、`kernels/cutlass`、`fundamentals/memory-hierarchy`、`overview/architecture` | SMEM 超过 99 KiB 时"申请被悄悄截断、写越界、没有错误码" | `cudaFuncSetAttribute` 返回 `cudaErrorInvalidValue`，超额启动报启动失败；问题是错误只在运行时出现 |
| `blackwell/thread-block-clusters` | Hopper 上 16 大小需要"可移植 cluster 大小"选项 | 属性是 `cudaFuncAttributeNonPortableClusterSizeAllowed` |
| `blackwell/nvfp4-deep-dive` | REAP = REbalanced Activation Pruning | Router-weighted Expert Activation Pruning |
| `compatibility/cluster-rewriting` | "三种思路" | 实际列了四种 |
| `compatibility/translating-tcgen05` | m64n64k16 → 16 条、m128n256k64 → 32 条 `mma.sync` | `mma.sync` 的 N 是 8，FP4 形状是 m16n8k64：分别为 32 条、256 条 |
| `compatibility/translating-tcgen05` | SM120 的 `mma.sync` 不带缩放，要在 MMA 后手工乘 | `sm_120a` 有块缩放 `mma.sync`，缩放因子作为寄存器操作数传入 |
| `kernels/marlin-and-friends` | "权重略大……其实反而略小" | Marlin 约 4.25 位/权重，比 NVFP4 的 4.5 位略小 |
| `kernels/inference-engines` | vLLM 参数 `--quantization fp4` | `modelopt_fp4` |

## 第二轮（2026-09-03）：清理与补充

### 修正

| 页面 | 原文 / 第一轮遗留 | 改为 |
| --- | --- | --- |
| `blackwell/sm100-vs-sm120` 故障表、`kernels/cutlass` 故障 2、`fundamentals/memory-hierarchy` 算例、`overview/architecture` 后果 3 | 仍写"cluster 悄悄降成 1"、"SMEM 越界悄悄写坏、没有报错" | 与第一轮结论一致：cluster 正常、SMEM 超限运行时报错 |
| `fundamentals/tensor-cores` | tile 形状表 `tcgen05.mma` 的 N 只到 128；`mma.sync` FP4 写成 m16n8k32 | N 到 256（单 CTA 步长 8，pair 步长 16），块缩放 kind 只有 M=128；FP4 是 `sm_120a` 的 m16n8k64 |
| `fundamentals/tensor-cores` | 按 2.1 GHz 反推每 SM 每周期 1.4 万次、差 7 倍 | NVIDIA 未公布频率；按额定算力反推 HGX 约 1.9 GHz、GB200 约 2.1 GHz，两者都得约 1.6 万次；差约 8 倍 |
| `fundamentals/number-formats` | `tcgen05` 里有专门的 NVFP4 → BF16 反量化指令 | 没有这条指令，反量化在块缩放 MMA 内部完成 |
| 全站 | PTX ISA 版本固定写 8.5；"主版本 8、次版本 5" | `tcgen05` 自 8.6 起，2026 年 9 月为 9.3（CUDA 13.3） |
| `blackwell/sm100-vs-sm120`、`fundamentals/memory-hierarchy`、`compatibility/smem-budget-management` | B100/B200 192 GB、8 TB/s；L2 H100 40 MB、B100 50 MB、B200 约 96 MB、SM120 约 16 MB | HGX B200 180 GB / 7.7 TB/s，GB200 内 186 GB / 8 TB/s；L2 H100 50 MB、B200 126 MB、RTX 5080 65 MB |
| `compatibility/smem-budget-management` | SM100 驱动预留 3 KiB、可用 225 KiB | Blackwell Tuning Guide：227 KB |
| `kernels/cutlass`、`kernels/deepgemm`、`overview/getting-started` | CUDA ≥ 12.4 / 12.5、CUTLASS 3.5/3.6 支持 SM100 | `sm_100a` 从 CUDA 12.8 起，CUTLASS 3.8 起；DeepGEMM SM100 要 12.9+ |
| `kernels/flashattention` | FA-3 能在 SM100 上跑、提速约 30 %；FA-Blackwell 开发中 | FA-3 建在 `wgmma` 上，Blackwell 上装不上；FA4（CuTe DSL）已发布 |
| `compatibility/runtime-detection` | TMEM 判断用 `sm_101` | `sm_103`（B300）；`sm_101` 在 CUDA 13 已改名 `sm_110`（Thor） |
| `fundamentals/cuda-pipeline` | `compute_100` 的 PTX 因含 `tcgen05` 不能跑在 12.0 上 | `tcgen05` 只在 `compute_100a` 里；不带后缀的 PTX 向前兼容 |
| `compatibility/translating-tcgen05` | `tcgen05.shift` 做"TMEM 布局变换" | 配合 weight-stationary MMA 的 lane 移位 |

### 新增

- `blackwell/tcgen05-and-tmem`：操作数来源、形状表（含块缩放 kind 无 M=64）、`.sp` / `.ws` 变体、无 C++ intrinsic；
  TMEM 分配的运行时规则（阻塞、递减、relinquish、dealloc、不进 occupancy）；`ld/st/cp` 形状；两套完成机制；
  CTA pair 的数据划分（B 共享、A/D 私有、配对规则、`.cta_group` 一致、TMA `.cta_group::2` 信号）。
- `fundamentals/cuda-pipeline`：`sm_100` / `sm_100a` / `sm_100f` / `sm_103a` 的真实特性集合；架构宏；
  Hopper 二进制在 B200 上的加载规则；版本门槛；CUDA 13.0 删掉的 `cudaDeviceProp` 字段。
- `blackwell/thread-block-clusters`：SM100 新增的 TMA `.cta_group`、gather4/scatter4、swizzle 原子性、
  FP4/FP6 解包类型、`st.bulk`、`clusterlaunchcontrol`、`setmaxnreg`。
- `fundamentals/number-formats`：kind 到 scale_vec 的映射；缩放因子在 TMEM 的布局规则；容器与打包；cvt 指令与头文件；
  cuBLASLt 枚举；INT4 不存在；FP8 累加精度 Hopper 与 Blackwell 的差异。
- `blackwell/nvfp4-deep-dive`：每张量 FP32 缩放是软件约定；SM100 侧硬件布局固定。
- `blackwell/sm100-vs-sm120`：别把 RTX 5080 的微基准当 B200；数据表稀疏/稠密。
- `kernels/deepgemm`、`kernels/flashattention`、`kernels/triton-and-transformerengine`、`kernels/cutlass`、
  `kernels/nvshmem-and-deepep`：各库在 B200 上的现状（DeepGEMM PR #112、FA4、Triton 3.3/3.4 与 Gluon、
  CUTLASS SM100 命名与 warp 分工、DeepEP）。
- 新页 `compatibility/hopper-to-sm100`：从 Hopper 迁到 B200 的坑与软件栈版本线。

## 已核实、无需改

- SM120 每 block 99 KB、每 SM 128 KB；SM100 每 block 227 KB、每 SM 228 KB。正文的 228 KiB 按每 SM 说，误差 1 KB，未改。
- `wgmma` 只在 `sm_90a`；`tcgen05` 由 PTX ISA 8.6 引入，最低目标 `sm_100a`。

## 存疑但未改

- 各页引用的 GitHub issue 编号、"40–70 %"一类经验数字未核实。
- `tcgen05` 页里 `ptxas` 报错的具体文案未核实。
- 正文里标了"第三方"或"未公布"的数字（B200 频率、L2 分区数、TMEM 与 occupancy 的关系、`mma.sync` 在 sm_100 的吞吐比例）没有官方来源。
