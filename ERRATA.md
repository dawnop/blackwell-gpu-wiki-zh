# 勘误表

译文对原文（`main` 分支）事实性错误做的修改。每处修改在页面里都有"（译注：……）"标注原文怎么写的。
核对依据：NVIDIA PTX ISA 文档、CUDA C++ Programming Guide、OCP Microscaling (MX) 规范、NVIDIA 公开产品规格。

## 贯穿多页的两类错误

### `wgmma.async` 不在 Blackwell 上

原文多处说 `wgmma.async` 能在 SM 10.0 和 SM 12.0 上跑（SM120 上"吞吐较低"）。
实际上 `wgmma` 是 `sm_90a` 专属：数据中心版 Blackwell 换成了 `tcgen05.mma`，
工作站版 Blackwell 只有 `mma.sync`（含 `sm_120a` 专属的块缩放 `mma.sync.kind::mxf4nvf4.block_scale` 等）。

改动页面：`fundamentals/tensor-cores`（特性表、CUTLASS 一节）、`blackwell/sm100-vs-sm120`（对照表）、
`blackwell/tcgen05-and-tmem`、`blackwell/index`（mermaid）、`overview/glossary`、`kernels/cutlass`、
`kernels/flashattention`、`kernels/triton-and-transformerengine`。

### `tcgen05` 指令的拼法和语义

原文的指令表和所有 PTX 示例用了不存在的拼法：按字节分配 TMEM、`tcgen05.wait`、
TMEM→SMEM 方向的 `tcgen05.cp`、`kind::nvf4` / `kind::f4`、把缩放因子当寄存器操作数等。按 PTX ISA 改写：

- `tcgen05.alloc` 按**列**分配（32–512 之间的 2 的幂），TMEM 地址写进 SMEM；整个 warp 一起执行。
- 没有 `tcgen05.wait`：`tcgen05.commit` 把已发射的异步操作挂到 **mbarrier** 上，用 `mbarrier.try_wait` 等。
- `tcgen05.cp` 只有 SMEM→TMEM 一个方向；TMEM 和寄存器之间用 `tcgen05.ld` / `tcgen05.st`。
- MMA 类型是 `kind::f16 / tf32 / f8f6f4 / i8 / mxf8f6f4 / mxf4 / mxf4nvf4`；NVFP4 走 `kind::mxf4nvf4.block_scale`，缩放因子放在 TMEM。
- `tcgen05.mma` 由**单个线程**发射，A、B 通过 SMEM 矩阵描述符给出。
- TMEM 是 128 lane × 512 列 × 32 位；原文的"默认/步长/复制/压缩"四种布局和"`tcgen05.shift` 做布局变换"不存在。
- CTA pair 模式：两个 CTA 各出一半 SMEM 操作数、各收一半 TMEM 结果，由 leader CTA 的一个线程发射一次。
- tile 上限：单 CTA M=128、N=256；CTA pair M=256、N=256（原文写 128×128 / 256×128）。
- 引入版本：PTX ISA 8.6（随 CUDA 12.8），原文写 8.4/8.5。

改动页面：`blackwell/tcgen05-and-tmem`（指令表、PTX 示例、TMEM 一节、CTA pair 一节、SM120 对照表）、
`fundamentals/tensor-cores`、`fundamentals/memory-hierarchy`、`fundamentals/cuda-pipeline`、
`compatibility/translating-tcgen05`（形状表、两段 PTX、缩放因子一节、伪代码）、`kernels/deepgemm`、
`blackwell/nvfp4-deep-dive`、`blackwell/sm100-vs-sm120`、`overview/glossary`。

## 单点错误

| 页面 | 原文 | 改为 |
| --- | --- | --- |
| `overview/architecture` | 计算能力"小数点后是主版本号" | 小数点前是主版本号 |
| `overview/glossary`、`fundamentals/cuda-pipeline` | `sm_NNf` = "forward-compatible" | NVIDIA 的叫法是 family-specific（家族专用） |
| `overview/glossary` | PCIe Gen4 / Gen5 每 lane 16 / 32 GB/s | 每 lane 16 / 32 GT/s（约 2 / 4 GB/s 每方向），x16 约 32 / 64 GB/s 每方向 |
| `fundamentals/memory-hierarchy` | m256n128k64 累加器 32 KB | m128n256 为 128 KB，m256n256 为 256 KB（32K 是元素个数） |
| `fundamentals/memory-hierarchy` | Hopper/SM100 的 L1+SMEM 合计 228 KiB | 合计 256 KB，其中最多 228 KB 可配给 SMEM |
| `fundamentals/number-formats` | FP32 是"1962 年的 IEEE 格式" | IEEE 754 是 1985 年 |
| `fundamentals/number-formats`、`kernels/deepgemm` | MX-FP4 缩放因子是 FP6 E3M2，每元素 4.19 位 | OCP MX 规范是 E8M0（8 位），每元素 4.25 位 |
| `fundamentals/tensor-cores` | 每个 Tensor Core 每周期发射一条 m128n128k64（1,048,576 次乘累加）；B100 144 个 SM、约 5 PFLOPs | 按 B200 公开峰值反推约每 SM 每周期 1.6 万次 FP4 乘累加；B200 148 个 SM、FP4 稠密约 9 PFLOPS |
| `fundamentals/tensor-cores` | RTX PRO 6000 FP4 约 125 TFLOPs，与 B100 相差 40 倍，一半来自 ISA | 按公开规格 RTX PRO 6000 约 2 PFLOPS（稠密）、B200 约 9 PFLOPS，差约 4–5 倍；折算到每 SM 每周期约 8 倍 |
| `blackwell/sm100-vs-sm120` | SMEM 超过 99 KiB 时"申请被悄悄截断、写越界、没有错误码" | `cudaFuncSetAttribute` 返回 `cudaErrorInvalidValue`，超额启动会报启动失败；问题是错误只在运行时出现 |
| `blackwell/thread-block-clusters` | Hopper 上 16 大小需要"可移植 cluster 大小"选项 | 属性是 `cudaFuncAttributeNonPortableClusterSizeAllowed`（非可移植） |
| `blackwell/nvfp4-deep-dive` | REAP = REbalanced Activation Pruning | Router-weighted Expert Activation Pruning（与缩写表、论文一致） |
| `compatibility/cluster-rewriting` | "三种思路" | 实际列了四种 |
| `compatibility/translating-tcgen05` | 形状表：m64n64k16 → 16 条、m128n256k64 → 32 条 `mma.sync` | `mma.sync` 的 N 是 8，FP4 形状是 m16n8k64：分别为 32 条、256 条 |
| `compatibility/translating-tcgen05` | SM120 的 `mma.sync` 不带缩放，要在 MMA 后手工乘 | `sm_120a` 有块缩放 `mma.sync`，缩放因子作为寄存器操作数传入 |
| `kernels/marlin-and-friends` | "权重略大……其实反而略小" | Marlin 约 4.25 位/权重，比 NVFP4 的 4.5 位略小，不算代价 |
| `kernels/inference-engines` | vLLM 参数 `--quantization fp4` | `modelopt_fp4`（与同页后文一致） |

## 存疑但未改

- 全站的核心前提"SM120 不支持 cluster > 1"、"SM120 每 block 99 KiB"未做独立核实，按原文保留。
- 各页引用的 GitHub issue 编号、库版本号、"40–70 %"一类经验数字未核实。
- `tcgen05` 页里 `ptxas` 报错的具体文案未核实。
- `interconnect/index` 说 TP 通信量"正比于序列长度而非词表大小"，表述含糊但不算错，未改。
