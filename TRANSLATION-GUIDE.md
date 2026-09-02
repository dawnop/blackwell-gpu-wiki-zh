# 翻译规范（zh 分支）

本分支把 `docs/` 下全部页面翻译成中文，原文在 `main` 分支。

## 读者

有 CUDA 基础、熟悉 Hopper 的工程师。默认读者已经知道：warp / CTA / SM /
共享内存 / TMA / `wgmma.async` / mbarrier / 线程块簇 / warp 特化 / 生产者-消费者流水线。
这些概念**不需要**解释；Blackwell 新引入的东西（TMEM、`tcgen05`、CTA pair、NVFP4）
按原文的解释翻，用大白话说清楚。

## 文风

- 简洁、直白、像中文技术作者自己写的，不要翻译腔。可以拆句、合句、调语序，
  但**不丢信息、不加私货、不删段落、不省略表格行**。
- 不说黑话：能用普通中文说清的就不堆术语。原文用了比喻或口语（"cliff"、"why doesn't this run?"）
  用同样轻松的中文表达。
- 原文事实有明显错误也照译，可以在句末加"（译注：……）"，但要克制。
- 中英文之间加一个空格；使用中文全角标点；数字、单位、版本号原样保留。
- 不在页面开头加"译者说明"，不在结尾加总结。

## 术语

### 保留英文不翻的

warp、CTA、SM、kernel（不译"核函数"）、SMEM、TMEM、TMA、`tcgen05`、`wgmma`、`mma.sync`、
mbarrier、PTX、SASS、cubin、fatbin、JIT、NVLink、NVSwitch、NVSwitch、PCIe、P2P、HBM、GDDR、
MoE、EP / TP / DP / PP、all-to-all、all-reduce、all-gather、GEMM、GEMV、tile、epilogue、
mainloop、pipeline（流水线也可）、dispatch / combine（MoE 术语）、CTA pair、cluster、
CUTLASS / CuTe、FlashAttention、FlashInfer、DeepGEMM、DeepEP、NVSHMEM、Marlin、Triton、
TransformerEngine、vLLM、SGLang、TensorRT-LLM、MLA、GQA、KV cache、FP4 / FP6 / FP8 / BF16 /
FP16 / TF32 / INT8 / INT4、NVFP4、MXFP4 / MXFP8、E4M3 / E5M2 / E2M1 / E8M0、
`sm_100a` / `sm_120` 之类的编译目标、所有指令名和 API 名。

### 固定译法

| 英文 | 中文 |
| --- | --- |
| compute capability | 计算能力（首次出现可标 CC） |
| datacenter Blackwell | 数据中心版 Blackwell |
| workstation / consumer Blackwell | 工作站/消费级 Blackwell |
| SM100 / SM120 | 原样保留 |
| shared memory | 共享内存（SMEM） |
| Tensor Memory | Tensor Memory（TMEM），首次出现可写"张量内存（TMEM）" |
| Tensor Core | Tensor Core |
| register / register file | 寄存器 / 寄存器堆 |
| accumulator | 累加器 |
| thread block cluster | 线程块簇（cluster） |
| warp specialization | warp 特化 |
| producer / consumer | 生产者 / 消费者 |
| issue（指令） | 发射 |
| commit / wait / barrier | 原样保留（描述性语境可说"屏障"） |
| occupancy | 占用率 |
| throughput / latency / bandwidth | 吞吐 / 延迟 / 带宽 |
| atomics | 原子操作 |
| fallback | 回退 |
| port / porting | 移植 |
| expert / routing / router | 专家 / 路由 / 路由器 |
| quantize / dequantize | 量化 / 反量化 |
| scale / block scale / scale factor | 缩放因子 / 块缩放因子 |
| microscaling (MX) | 微缩放（MX） |
| inference engine | 推理引擎 |
| the 99 KiB cliff / ceiling | 99 KiB 断崖 / 上限 |
| interconnect | 互连 |
| topology | 拓扑 |
| fabric | fabric（首次可写"互连 fabric"） |
| prefill / decode | prefill / decode（可加"预填充 / 解码"） |
| open-weight model | 开放权重模型 |
| frontier lab | 前沿实验室 |
| case study | 案例分析 |
| compatibility pattern | 兼容性套路 / 兼容方案 |
| glossary / abbreviations / bibliography | 术语表 / 缩写表 / 参考文献 |

## 格式（必须严格保持）

- 文件名、目录结构、相对链接路径、图片路径一律不动。
- 所有 Markdown 结构原样保留：标题层级、列表、表格列数、粗体、行内代码、分隔线。
- 代码块：代码本身不动；代码块里的**注释**翻成中文。
- mermaid 图：结构不动，节点/边上的文字翻成中文。
- 表格里的指令、数字、型号原样保留，只翻描述性文字。
- 链接文字翻成中文，链接目标不变。
