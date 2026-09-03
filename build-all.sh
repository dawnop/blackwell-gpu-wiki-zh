#!/usr/bin/env bash
# 构建中英双语站点：site/ 是中文（docs/），site/en/ 是英文（en-src/docs/）。
set -euo pipefail
cd "$(dirname "$0")"
MKDOCS="${MKDOCS:-mkdocs}"
rm -rf site
mkdir -p en-src/docs/javascripts
cp docs/javascripts/lang-switch.js en-src/docs/javascripts/
"$MKDOCS" build --strict
"$MKDOCS" build --strict -f mkdocs.en.yml
echo "built: site/ (zh) + site/en/ (en)"
