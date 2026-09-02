// 语言切换时停在当前页面：中英两套站点的页面路径完全一致，只差一个 en/ 前缀。
(function () {
  var links = document.querySelectorAll("a.md-select__link[hreflang]");
  if (!links.length) return;
  var zh = null, en = null;
  links.forEach(function (a) {
    if (a.getAttribute("hreflang") === "zh") zh = a;
    if (a.getAttribute("hreflang") === "en") en = a;
  });
  if (!zh || !en) return;
  var base = new URL(zh.href).pathname;          // 例如 /blackwell-gpu-wiki-zh/
  var enBase = base + "en/";
  var path = location.pathname;
  var rel;
  if (path.indexOf(enBase) === 0) rel = path.slice(enBase.length);
  else if (path.indexOf(base) === 0) rel = path.slice(base.length);
  else return;
  zh.href = base + rel + location.hash;
  en.href = enBase + rel + location.hash;
})();
