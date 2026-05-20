# Issue tracker: GitHub

Issues 和 PRD 使用 GitHub Issues 管理，所有操作通过 `gh` CLI 进行。

## 操作约定

- **创建 issue**: `gh issue create --title "..." --body "..."`，多行正文使用 heredoc。
- **查看 issue**: `gh issue view <number> --comments`，用 `jq` 过滤评论并获取标签。
- **列出 issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，根据需要添加 `--label` 和 `--state` 过滤。
- **评论 issue**: `gh issue comment <number> --body "..."`
- **添加/移除标签**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭 issue**: `gh issue close <number> --comment "..."`

`gh` 在仓库内运行时会自动从 `git remote -v` 推导仓库名。

## 技能说明

- "发布到 issue tracker" → 创建一个 GitHub issue
- "获取相关的 issue" → 运行 `gh issue view <number> --comments`
