#!/usr/bin/env bash
# 构建中英双语站点：site/ 是中文（zh 分支），site/en/ 是英文原文（main 分支，可用 EN_REF 指定）。
set -euo pipefail
cd "$(dirname "$0")"
MKDOCS="${MKDOCS:-mkdocs}"
rm -rf en-src site
mkdir -p en-src
git archive "${EN_REF:-main}" docs mkdocs.yml | tar -x -C en-src
mkdir -p en-src/docs/javascripts
cp docs/javascripts/lang-switch.js en-src/docs/javascripts/
"$MKDOCS" build --strict
"$MKDOCS" build --strict -f mkdocs.en.yml
echo "built: site/ (zh) + site/en/ (en)"
