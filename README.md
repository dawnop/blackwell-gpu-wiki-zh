# Blackwell GPU Wiki（中文版）

[0xSero/blackwell-gpu-wiki](https://github.com/0xSero/blackwell-gpu-wiki) 的中文翻译，供本地阅读。
原文讲的是 NVIDIA Blackwell 一代里 **SM 10.0**（数据中心版）和 **SM 12.0**（工作站/消费级）
两条 ISA 的分野：SM100 与 SM120 的差异、NVFP4、`tcgen05`/TMEM、MoE 推理、内核库现状和兼容方案。

- `main` 分支：上游英文原文。
- `zh` 分支：中文译文，页面路径与原文一一对应。翻译约定见 `TRANSLATION-GUIDE.md`；
  译文修正了原文的一批事实性错误，清单见 `ERRATA.md`，页面里对应位置有"译注"。

在线阅读（GitHub Pages，随 `zh` 分支自动部署）：<https://dawnop.github.io/blackwell-gpu-wiki-zh/>

英文原站：<https://blackwell-gpu-wiki.pages.dev/>（2026 年 9 月时已打不开）；上游仓库：<https://github.com/0xSero/blackwell-gpu-wiki>

## 已提给上游的修正

译文里修正的错误已按主题提给上游仓库，页面里的"译注"对应下列条目；上游合并后译注会随之删除。

| 上游链接 | 内容 |
| --- | --- |
| [issue #1](https://github.com/0xSero/blackwell-gpu-wiki/issues/1) / [PR #3](https://github.com/0xSero/blackwell-gpu-wiki/pull/3) | SM120 支持线程块簇（最多 8），"cluster 只能为 1"及其推论不成立 |
| [issue #2](https://github.com/0xSero/blackwell-gpu-wiki/issues/2) | Tensor Core 页的性能数字与 NVIDIA 公开规格不符 |
| [PR #4](https://github.com/0xSero/blackwell-gpu-wiki/pull/4) | `wgmma.async` 只在 `sm_90a`，Blackwell 两个分支都没有 |
| [PR #5](https://github.com/0xSero/blackwell-gpu-wiki/pull/5) | `tcgen05` 指令拼法、TMEM 组织、CTA pair、tile 上限按 PTX ISA 重写 |
| [PR #6](https://github.com/0xSero/blackwell-gpu-wiki/pull/6) | MX-FP4 缩放因子是 E8M0；IEEE 754 是 1985 年 |
| [PR #7](https://github.com/0xSero/blackwell-gpu-wiki/pull/7) | `mma.sync` 条数表、块缩放 FP4 写法、改写页的 PTX 示例 |
| [PR #8](https://github.com/0xSero/blackwell-gpu-wiki/pull/8) | 主次版本号、`sm_NNf`、PCIe 单位、REAP、Marlin、vLLM 参数、SMEM 超限行为等杂项 |

完整清单见 `ERRATA.md`。

## 本地阅读

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

然后打开 <http://127.0.0.1:8000/>。

## 构建静态站

```bash
mkdocs build
# 输出在 ./site
```

## 协议

原文 MIT，译文沿用 MIT。见 [LICENSE](LICENSE)。
