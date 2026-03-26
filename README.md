# deep_research_agent
Regularly capture new domain blogs every day and output insight analysis reports.

## 在线阅读报告 (GitHub Pages)

仓库根目录的 [`index.html`](index.html) 会列出 `ai-daily-insight/content` 与 `cloud-daily-insight/content` 下的 Markdown，并渲染阅读。依赖同级的 [`js/marked.min.js`](js/marked.min.js)、[`js/purify.min.js`](js/purify.min.js) 与 [`css/github-markdown.min.css`](css/github-markdown.min.css)（均随仓库提供，避免 CDN 被拦截后出现无样式「记事本」或 `marked.parse` 报错）。

1. 在 GitHub 仓库 **Settings → Pages** 中，**Source** 选 **Deploy from a branch**，分支选默认分支（如 `main`），目录选 **/ (root)**，保存。
2. 站点地址一般为：`https://fooljohnny.github.io/deep_research_agent/` 或 `.../index.html`。
3. 若默认分支不是 `main`，打开页面时在 URL 后加参数，例如：  
   `?branch=main`（或与你的默认分支一致）。

列表数据来自 **GitHub Contents API**（匿名有速率限制）；正文来自 **raw.githubusercontent.com**。`index.html` 对每次访问使用 **禁用缓存** 与 **cache-bust** 参数，便于在报告每日更新后打开页面即看到最新列表与正文。

报告里的趋势图等使用相对路径（如 `charts/*.png`）；在 `github.io` 上会自动改写为对应分支下的 **raw** 图片地址，避免 broken image。

### 若控制台仍报 `cdn.jsdelivr.net` / `ERR_TUNNEL_CONNECTION_FAILED`

当前 `index.html` **不再引用** jsDelivr；样式与脚本均来自本站 `css/`、`js/`。若 F12 里仍出现对 `cdn.jsdelivr.net/.../github-markdown.min.css` 的请求，说明浏览器或中间代理还在用**旧版页面**：

1. 确认 **GitHub Pages** 的发布分支已包含最新的 `index.html` 与 `css/github-markdown.min.css`（若用 `gh-pages` 分支部署，需已合并/推送该分支）。
2. 对 `index.html` 做一次 **强制刷新**（Ctrl+Shift+R 或清空本站缓存）后再打开。
