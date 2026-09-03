# Blackwell GPU Wiki（中英双语）

讲 NVIDIA Blackwell 一代里 **SM 10.0**（数据中心版 B200 / GB200）和 **SM 12.0**（工作站/消费级 RTX PRO、RTX 50）
两条 ISA 的分野：`tcgen05` / TMEM、CTA pair、NVFP4、cluster 与 TMA、SMEM 预算、MoE 推理、内核库现状、
两个方向的移植（SM100 → SM120，以及 Hopper → B200）。

在线阅读（GitHub Pages，随 `zh` 分支自动部署）：<https://dawnop.github.io/blackwell-gpu-wiki-zh/>
页面右上角可以中英切换，英文挂在 `/en/` 下，切换时停留在同一页。

中英文都以本仓库为准、同步维护：事实性修正和新增内容两种语言一起改。
与原始出处相比改了什么，见 `ERRATA.md`。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `docs/` | 中文正文 |
| `en-src/docs/` | 英文正文 |
| `mkdocs.yml` / `mkdocs.en.yml` / `en-src/mkdocs.yml` | 两套站点配置 |
| `build-all.sh` | 一次构建两种语言到 `site/` 和 `site/en/` |
| `ERRATA.md` | 相对原始出处的修正与新增清单 |
| `TRANSLATION-GUIDE.md` | 中文文风与术语约定 |
| `main` 分支 | 原始出处的英文快照，只读，不再更新 |

## 本地阅读

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve                       # 只有中文
```

构建双语静态站：

```bash
./build-all.sh
python3 -m http.server -d site 8000
# 打开 http://127.0.0.1:8000/blackwell-gpu-wiki-zh/
```

## 协议

MIT。见 [LICENSE](LICENSE)。

## 致谢

正文最初翻译自 [0xSero/blackwell-gpu-wiki](https://github.com/0xSero/blackwell-gpu-wiki)（MIT）。
翻译过程中核对出的错误曾以 issue #1、#2 和 PR #3 到 #8 的形式提交给该仓库；此后本仓库独立维护，
不再跟随上游。原始快照保留在 `main` 分支。
