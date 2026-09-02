# NVFP4 深入解析

NVIDIA 版的 OCP MX-FP4。它是唯一一个在 Blackwell 两个分支上真正表现一致的 Blackwell 专属格式，也是让 478B 参数模型塞进 4 张 96 GB 显存卡的秘诀。

## 格式的精确定义

一个 NVFP4 张量以 **16 个 4 位值为一块**编码，每块附带**一个 FP8（E4M3）缩放因子**。

```
块布局（内存中）：
+-------------------------------------+----------+
| 16 × 4 位值（打包成 8 字节）           | 缩放因子  |
+-------------------------------------+----------+
                                      8 位

合计：每 16 个值占 9 字节 = 每个值 4.5 位
```

每个 4 位值是 **E2M1**：1 位符号 + 2 位指数 + 1 位尾数，可表示的值为 {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}（全 1 的位模式表示 NaN）。

块 `b` 中第 `i` 个元素的实际数值是：

```
value(i, b) = decode_e2m1(packed[b][i]) × decode_e4m3(scale[b])
```

块级缩放因子给了这个格式有效的动态范围：每块缩放因子从 10⁻⁴ 到 10² 乘以 E2M1 的范围，得到一个横跨多个数量级的可用工作范围。

## NVFP4 与 MX-FP4：关键差异

OCP MX-FP4 标准结构类似，但参数不同：

| | MX-FP4（OCP） | NVFP4（NVIDIA） |
| --- | --- | --- |
| 块大小 | 32 个元素 | **16 个元素** |
| 缩放因子类型 | FP6（E3M2） | **FP8（E4M3）** |
| 每元素有效位数 | ~4.19 | ~4.50 |
| 采用者 | AMD、Intel、ARM、NVIDIA | NVIDIA |

NVFP4 用**略多一点的存储**（每元素 4.5 位对 4.19 位，约 7 % 的开销）换来**两个实际好处**：

1. **更小的块**意味着数值分布不均匀的张量（比如某些通道有离群值）能拿到更贴合的缩放因子。经验数据：在相同有效码率下，大多数 LLM 基准上比 MX-FP4 的困惑度改善约 0.3–0.5。
2. **FP8 缩放因子**的精度大约是 FP6 缩放因子的 16 倍，减少了缩放因子本身量化带来的误差。对权重幅值极端的层来说这很重要。

两种格式都能在 Blackwell 的 Tensor Core（第五代）上原生运行。Hopper 硬件通过 FP8 Tensor Core 模拟它们，吞吐会打折。

## 省下的显存，用数字说话

以一个 4780 亿参数的 LLM 为例：

| 格式 | 权重总存储 |
| --- | ---: |
| FP32 | ~1.9 TB |
| BF16 / FP16 | ~960 GB |
| FP8 | ~480 GB |
| MX-FP4（4.19 位/元素） | ~250 GB |
| **NVFP4（4.5 位/元素）** | **~270 GB** |

实际上，NVFP4 把一个 BF16 模型压到大约**原来的 28 %**，原本需要 8 倍显存的模型也能装下。对工作站版 Blackwell 来说，96 GB 的卡 × 4 = 384 GB 总显存，可以放下约 700B 参数的权重，还给 KV cache 留有余地。

## Tensor Core 路径

在 Blackwell 上，一个 NVFP4 GEMM 会编译成以 NVFP4 为输入、输出 FP32（或 BF16、FP8）的 MMA 指令。反量化发生在 **Tensor Core 内部**——MMA 之前不需要单独的"反量化 kernel"。

```ptx
// 数据中心版 Blackwell（SM100）—— 使用 tcgen05
tcgen05.mma.cta_group::1.kind::nvf4
    [%tmem_d],              // TMEM 里的 FP32 累加器
    [%smem_a],              // NVFP4 操作数 A
    [%smem_b],              // NVFP4 操作数 B
    %scale_a, %scale_b;     // E4M3 缩放因子寄存器

// 工作站版 Blackwell（SM120）—— 使用 mma.sync 链
// （对更大逻辑 tile 中的每个 m16n8k32 子 tile：）
mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32
    {%rd0, %rd1, %rd2, %rd3},     // FP32 累加器
    {%ra0, %ra1},                  // NVFP4 操作数 A（缩放因子隐式处理）
    {%rb0, %rb1},                  // NVFP4 操作数 B
    {%rc0, %rc1, %rc2, %rc3};      // FP32 输入累加器
```

两条路径用的是**同一套 Tensor Core 硬件**，区别只在发射指令。SM100 的 `tcgen05.mma.kind::nvf4` 异步地把更大的 tile 发射到 TMEM；SM120 的 `mma.sync.m16n8k32.f32.e2m1.e2m1.f32` 同步地把更小的 tile 发射到寄存器。

## 缩放因子布局问题

尽管 NVFP4 在 Blackwell 的两个分支上都能原生运行，**缩放因子布局不兼容**却是反复出现的 bug 来源。

格式定义了一个 NVFP4 块"是什么"——值 + 缩放因子——但没有完全规定缩放因子在内存中如何与值交错。目前有好几种布局在实际使用：

| 布局 | 缩放因子放在哪 |
| --- | --- |
| **块交错** | 每 16 个值后面跟一个：`[16v, scale, 16v, scale, ...]` |
| **值后接缩放因子** | 先放全部值，再放全部缩放因子：`[v0..vN, s0..sM]` |
| **按 K 维缩放** | K 维每个块一个缩放因子，向量化方式不同 |
| **tile 主序缩放因子** | 缩放因子打包成一个单独的张量，有自己的 tile 结构 |

CUTLASS 的 NVFP4 GEMM 模板每个模板假定一种特定布局。DeepGEMM 用的是另一种。FlashInfer 又是一种。**把以某种布局存储的权重文件加载进期望另一种布局的 kernel，产出的是静默的垃圾数据。**

"MX-FP4 还是 NVFP4"的混淆让事情更复杂：一个 kernel 可能标着"MX-FP4"，实际却期望 NVFP4 布局（因为开发者拷了一份 NVFP4 参考实现然后改了名）。缩放因子明明是 FP8 却按 FP6 来读，每一块都会被读坏。

当一个"本该能跑"的模型在工作站版 Blackwell 上输出乱码时，缩放因子布局不匹配是前三大原因之一（另外两个是 SMEM 断崖和 EP 方案）。

## 量化流程

把模型转成 NVFP4 的过程是：

1. 在校准数据上**分析**权重和激活值，确定每块的缩放因子
2. 量化前**施加按通道或按张量的缩放**，让各处的数值范围均衡
3. 在每块内**舍入**到最近的可表示 E2M1 值
4. **可选地微调**（"量化感知训练"）以找回精度

生产级流程（NVIDIA 的 TransformerEngine、Hugging Face 的 transformers + bitsandbytes、各厂商自己的工具）会把这些自动化。输出是一个 HuggingFace 模型文件，权重按 NVFP4 布局存放，每块的缩放因子作为单独的张量存放。

一个典型的 Hugging Face 模型目录：

```
my-model-NVFP4/
├── config.json
├── tokenizer.json
├── chat_template.jinja
├── model.safetensors           # NVFP4 打包的权重
├── scales.safetensors          # FP8 E4M3 缩放因子
├── quantization_config.json    # block_size: 16, scale_dtype: fp8_e4m3
└── ...
```

`quantization_config.json` 是判断该模型文件用哪种布局的唯一依据。加载器如果无视它、假定了错误的布局，就会把所有东西都解码错。

## 与其他低比特格式的质量对比

在标准 LLM 基准上的经验数据：

| 格式 | 相对 BF16 的困惑度损失 |
| --- | ---: |
| FP8（E4M3） | ~0.0 |
| MX-FP4 | ~0.5–1.0 |
| **NVFP4** | **~0.3–0.5** |
| INT4（按通道缩放，如 AWQ） | ~0.5–1.5 |
| INT4（按块缩放，如 GPTQ） | ~0.3–0.7 |

NVFP4 与最好的 INT4 方案不相上下，略优于 MX-FP4。它真正的优势在**推理的简单性**：格式由 Tensor Core 原生支持，反量化不花钱；而 INT4 方案需要一个单独的反量化 kernel，要付出吞吐代价。

## REAP 与 NVFP4 联用

REAP（REbalanced Activation Pruning）是一种 MoE 专用的剪枝技术，会把整个专家从模型里去掉。和 NVFP4 量化结合，REAP 能产出质量损失极小的紧凑模型：

- 原始 GLM-5.1：744B 参数，BF16 约 1.5 TB
- REAP-160（256 个专家保留 160 个）：478B 参数，BF16 约 960 GB
- REAP-160 + NVFP4：478B 参数，NVFP4 约 270 GB

270 GB 的版本能装进 4 张 96 GB 的工作站版 Blackwell 卡，还剩约 50 GB 给 KV cache——足够 200k token 的上下文。

正是这个组合，让"把工作站版 Blackwell 当 MoE 平台"这整个话题有了意义。没有 NVFP4，这些模型装不下；没有 MoE 专家剪枝，同样装不下。两者合在一起，让一类 18 个月前还不可能的部署方式成为现实。

## 自测

你应该能回答：

- NVFP4 的块大小和缩放因子类型是什么？
- NVFP4 为什么用比 MX-FP4 更小的块？
- NVFP4 在工作站版 Blackwell 上能原生运行吗？在 Hopper 上呢？
- NVFP4 权重产出静默垃圾数据时，最常见的 bug 是什么？
- NVFP4 每个元素大约占多少位？

## 另见

- [`fundamentals/number-formats`](../fundamentals/number-formats.md) —— 数值格式总览
- [`tcgen05-and-tmem`](tcgen05-and-tmem.md) —— SM100 的 NVFP4 路径
- [`kernels/cutlass`](../kernels/cutlass.md) 与 [`kernels/deepgemm`](../kernels/deepgemm.md) —— 各库各自的 NVFP4 路径
- *Open Compute Project Microscaling Format Specification*
- *NVIDIA Blackwell Architecture Whitepaper*，"FP4 Tensor Cores"一节
- HuggingFace 博客："NVFP4 quantization for inference"
