# Blackwell GPU Wiki（中文版）

[0xSero/blackwell-gpu-wiki](https://github.com/0xSero/blackwell-gpu-wiki) 的中文翻译，供本地阅读。
原文讲的是 NVIDIA Blackwell 一代里 **SM 10.0**（数据中心版）和 **SM 12.0**（工作站/消费级）
两条 ISA 的分野：SM100 与 SM120 的差异、NVFP4、`tcgen05`/TMEM、MoE 推理、内核库现状和兼容方案。

- `main` 分支：上游英文原文。
- `zh` 分支：中文译文，页面路径与原文一一对应。翻译约定见 `TRANSLATION-GUIDE.md`；
  译文修正了原文的一批事实性错误，清单见 `ERRATA.md`，页面里对应位置有"译注"。

在线阅读（GitHub Pages，随 `zh` 分支自动部署）：<https://dawnop.github.io/blackwell-gpu-wiki-zh/>

英文原站：<https://blackwell-gpu-wiki.pages.dev/>（2026 年 9 月时已打不开）；上游仓库：<https://github.com/0xSero/blackwell-gpu-wiki>

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
