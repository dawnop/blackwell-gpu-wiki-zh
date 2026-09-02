# CUDA 编译流水线

一个 `.cu` 源文件是怎么一步步变成 SM 上执行的指令的。理解这条流水线很关键，因为大多数 SM100/SM120 不兼容问题，都是在*这条流水线的某个环节*上暴露出来的。

## 流水线一览

```mermaid
flowchart LR
    SRC[".cu 源文件"] --> NVCC["nvcc"]
    NVCC --> HOSTOBJ["host .o<br/>（CPU 侧 C++）"]
    NVCC --> PTX[".ptx<br/>虚拟 ISA"]
    PTX --> PTXAS["ptxas"]
    PTXAS --> CUBIN["cubin<br/>sm_NN 的 SASS"]
    CUBIN --> FATBIN[".fatbin<br/>多架构容器"]
    HOSTOBJ --> EXE["host 可执行文件"]
    FATBIN --> EXE
    EXE -.运行时.-> DRIVER["驱动"]
    DRIVER --> JIT["没有匹配的 cubin 时<br/>JIT 编译 PTX"]
    DRIVER --> LAUNCH["在 SM 上启动 SASS"]
```

关键在于，`.cu` 源码要经过**两步编译**：一步高层的（`nvcc` / NVCC 的 PTX 后端），一步低层的（`ptxas`）。每一步都可能因为不同的原因成功或失败。

## 第 1 步：nvcc → PTX

`nvcc` 是一个驱动式编译器，它会：

1. 把 `.cu` 源码拆成 host 代码（C++）和 device 代码（CUDA C++）
2. 用 host 侧的 C++ 编译器（gcc/clang/cl）编译 host 代码
3. 用自己的前端把 device 代码编译成 PTX

PTX（**Parallel Thread eXecution**）是 NVIDIA 的**虚拟 ISA**——一种（在一定范围内）与具体架构无关的中间表示。可以把它理解成专门给 GPU 代码用的 LLVM IR。

用 `--gpu-architecture`（或 `-arch`）指定目标 PTX 版本：

```
nvcc -arch=compute_100 ...   # 面向计算能力 10.0 的 PTX
nvcc -arch=compute_120 ...   # 面向计算能力 12.0 的 PTX
```

PTX **在同一大版本内向前兼容**：面向 `compute_70` 的 PTX 可以被 JIT 编译后跑在任何更新的架构上（8.0、9.0、10.0、12.0）。面向 `compute_100` 的 PTX 只能跑在 10.0 及之后的 1x.x 架构上，但**不能**跑在 12.0 上（因为 10.0 引入了 `tcgen05` 这类 12.0 不支持的指令——它们是 1x 家族里的不同"分支"）。

### 这一步会出什么问题

- **指令缺失**：源码用了目标 PTX 版本里没有的内建函数（`__hadd2`、`__nvvm_reflect`、`cp.async.bulk.tensor`）。编译报错。
- **架构专属内建函数用错了目标**：源码用了 `tcgen05.mma`，却用 `-arch=compute_120` 编译。编译报错："instruction not supported in this PTX version"。
- **源码里的向前兼容写错了**：代码用 `__CUDA_ARCH__` 宏把数据中心专属的路径隔开，但条件写错了。这种情况往往能生成可用的 PTX，却在运行时挂掉。

## 第 2 步：ptxas 把 PTX 变成 SASS

`ptxas`（PTX 汇编器）把 PTX 降级成 **SASS**——真正在 SM 上执行的、按架构区分的机器码。SASS 的文档很少；你主要通过 `nvdisasm` 或 `cuobjdump --dump-sass` 跟它打交道。

用 `--gpu-code`（或 `-code`）指定目标 SASS 架构：

```
ptxas --gpu-name=sm_100  ...
ptxas --gpu-name=sm_120  ...
```

或者和 `nvcc` 一起用：

```
nvcc -gencode arch=compute_100,code=sm_100   ...
nvcc -gencode arch=compute_120,code=sm_120   ...
```

`-gencode` 这种写法会为指定架构生成 SASS，*同时*嵌入 PTX（如果二进制跑在一个不在 gencode 列表里的未来架构上，PTX 可以被 JIT 编译）。

### `a` 和 `f` 后缀

NVIDIA 引入了两个后缀来管理架构专属特性：

| 后缀 | 含义 | 示例 |
| --- | --- | --- |
| （无） | 该架构的"可移植"子集 | `sm_100` |
| `a` | "架构专属加速"——使用不可移植的特性。代码*只能*跑在这一个架构上。 | `sm_100a` |
| `f` | "向前兼容"——只允许使用在本架构以及同一大版本后续架构上都会存在的指令 | `sm_120f` |

实际使用中：

- **`sm_100a`** 允许 `tcgen05` 指令、MNNVL fabric 调用以及其他 GB100 专属特性。编出来的 SASS 只能跑在 10.0 设备上。
- **`sm_100`** 是更保守的目标，不含上述特性。
- **`sm_120a`** 允许 GB202 专属特性（例如只在消费级 Blackwell 上才有的某些 Tensor Core 变体），只能跑在 12.0 上。
- **`sm_120f`** 是"面向未来"的子集，能跑在 `sm_120` 和之后所有 12.x 架构上。适合要覆盖一大批消费级 Blackwell 型号的库。

NVIDIA 自家的库里就能看到后缀的选择：

- CUTLASS 的 Blackwell 模板用 **`sm_100a`**，因为它们需要 `tcgen05`
- 移植到工作站版时会用 **`sm_120`** 或 **`sm_120f`**，不带 `tcgen05`

### 这一步会出什么问题

- **指令不可用**：PTX 里有 `tcgen05.mma`，但 `--gpu-name=sm_120`。ptxas 报错。
- **SMEM 超额分配**：PTX 申请的 SMEM 超过了目标架构的容量。ptxas 可能只给警告，但生成的二进制会在运行时失败。
- **寄存器堆溢出**：PTX 每线程需要的寄存器数超过目标架构支持的上限。ptxas 会把寄存器溢出（spill）到局部内存（一块由 HBM 承载的线程私有区域），很慢。
- **cluster 形状不支持**：PTX 声明了 `.cluster_dim 2,1,1`，但目标架构不支持 cluster，或者支持的最大尺寸更小。

## 第 3 步：cubin 与 fatbin

**cubin** 是针对某一个架构编译出来的二进制：它包含 `sm_NN` 的 SASS，还可以选择性地嵌入 PTX。

**fatbin** 是一个容器，装着多个架构的 cubin，外加可选的 PTX。给 `nvcc` 传多个 `-gencode` 就会得到 fatbin：

```
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100a \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     ...
```

最后一行（`code=compute_120`）嵌入的是 `compute_120` 的 **PTX**，在没有匹配 cubin 的情况下，驱动可以在加载时把它 JIT 编译成 SASS。

### 查看 fatbin 内容

```bash
cuobjdump --list-elf  myapp           # 看里面有哪些架构
cuobjdump --dump-elf  myapp           # 导出 SASS
cuobjdump --dump-ptx  myapp           # 导出嵌入的 PTX
```

实践中就是靠这个发现，某个预编译的库只面向 `sm_100a` 而不是 `sm_120`——它的 fatbin 里只有 `sm_100a` 的 cubin。

## 第 4 步：运行时——驱动、JIT、启动

CUDA 程序加载一个 kernel 时：

1. 驱动在 fatbin 里查找这个 kernel。
2. 如果存在与设备架构匹配的 cubin，驱动直接加载它。
3. 如果没有匹配的 cubin，驱动去找嵌入的 PTX。找到了就 JIT 编译。
4. 两者都没有，kernel 加载失败，报类似 `CUDA error: no kernel image is available for execution on the device` 的错误。

SM120 去加载一个只有 SM100 的 fatbin，就会在这一步失败。**这是用户第一次撞上 SM100/SM120 分裂的地方。**

驱动会把 JIT 结果跨运行缓存起来，所以第一次启动慢，之后就快了。Linux 上缓存位于 `~/.nv/ComputeCache/`。

## 实例：追查一个 `tcgen05` 错误

假设你 `pip install` 了一个 kernel 库，在 SM120 的卡上跑，得到：

```
CUDA error: no kernel image is available for execution on the device
```

沿着流水线排查：

1. **找到 .so**：定位包含这个 kernel 的共享库。
2. **查看 fatbin**：`cuobjdump --list-elf libfoo.so | grep arch`。输出：
   ```
   arch = sm_90
   arch = sm_100a
   ```
   没有 `sm_120` 的 cubin → 这就是加载失败的原因。
3. **检查有没有 PTX 回退**：`cuobjdump --dump-ptx libfoo.so | head`。如果有 PTX，看它的目标：
   ```
   .version 8.5
   .target sm_100a
   ```
   目标是 `sm_100a` → JIT 到 `sm_120` 同样会失败（1x 家族里的不同大版本分支）。
4. **读 PTX**：搜 `tcgen05`。如果有：
   ```
   tcgen05.alloc.cta_group::1 %rd5, 16384;
   ```
   确认了：这个 kernel 用了数据中心专属指令。没有任何自动回退。

到这一步，解决办法只有：

- 用 `-arch=compute_120` 从源码重新编译，并换一套（面向 SM120 的）实现
- 换一个支持 SM120 的 kernel 库
- 换到数据中心版 Blackwell 的卡上跑

你没法"强行"让 SM100 的 kernel 跑在 SM120 上——两者在机器指令层面就是不同的操作。

## 编译选项速查

想弄清楚发生了什么，最有用的几个 nvcc 选项：

```bash
nvcc -keep -keep-dir build/intermediate ...   # 保留中间文件
nvcc --ptxas-options=-v ...                   # ptxas 详细输出：SMEM/寄存器用量
nvcc --resource-usage ...                     # 打印每个 kernel 的寄存器/SMEM 用量
nvcc -G ...                                   # 生成 device 侧调试信息
nvcc -lineinfo ...                            # SASS 里带源码行号信息（给 ncu 用）
```

查看编译产物：

```bash
cuobjdump --list-elf libfoo.so                # 这个 fatbin 里有哪些架构
cuobjdump --dump-elf libfoo.so > sass.txt     # 导出 SASS
cuobjdump --dump-ptx libfoo.so > ptx.txt      # 导出 PTX
nvdisasm sass.txt                              # 反汇编 SASS
```

## 自测

你应该能回答：

- PTX 和 SASS 有什么区别？
- `sm_100`、`sm_100a`、`sm_100f` 有什么区别？
- 一个 kernel 在 H100 上能跑、在 B100 上不能跑，最可能的问题是什么？
- 一个 kernel 在 B100 上能跑、在 RTX 5090 上不能跑，最可能的问题是什么？
- 驱动为什么要有 JIT 这条路？

## 另见

- [`tensor-cores`](tensor-cores.md)——`mma.sync` 和 `tcgen05.mma` 到底是什么
- [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md)——PTX 和 SASS 层面的具体差异
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md)——怎么把 SM100 的 PTX 降级成 SM120 的 PTX
- NVIDIA *PTX ISA* 规范（截至 2026 年为 8.5）
- NVIDIA *CUDA Binary Utilities* 文档
