# deep_research_agent
Regularly capture new domain blogs every day and output insight analysis reports.

## 在线阅读报告 (GitHub Pages)

仓库根目录的 [`index.html`](index.html) 会列出 `ai-daily-insight/content` 与 `cloud-daily-insight/content` 下的 Markdown，并渲染阅读。

1. 在 GitHub 仓库 **Settings → Pages** 中，**Source** 选 **Deploy from a branch**，分支选默认分支（如 `main`），目录选 **/ (root)**，保存。
2. 站点地址一般为：`https://fooljohnny.github.io/deep_research_agent/` 或 `.../index.html`。
3. 若默认分支不是 `main`，打开页面时在 URL 后加参数，例如：  
   `?branch=main`（或与你的默认分支一致）。

列表数据来自 **GitHub Contents API**（匿名有速率限制）；正文来自 **raw.githubusercontent.com**。`index.html` 对每次访问使用 **禁用缓存** 与 **cache-bust** 参数，便于在报告每日更新后打开页面即看到最新列表与正文。

报告里的趋势图等使用相对路径（如 `charts/*.png`）；在 `github.io` 上会自动改写为对应分支下的 **raw** 图片地址，避免 broken image。
