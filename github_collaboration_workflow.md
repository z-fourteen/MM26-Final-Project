# GitHub 协作开发规范与工作流

本文档适用于大学课程项目、实验室项目和中小型工程项目，目标是让团队成员在 GitHub 上规范协作、减少冲突、降低误操作风险，并保证代码可以稳定合并与发布。

## 1. 分支管理规范

### 1.1 推荐协作模型

本项目推荐使用简化 Git Flow：

- `main`：稳定发布分支，只保存已经验证通过、可以展示或交付的版本。
- `dev`：日常集成分支，所有功能分支完成后先合并到 `dev`。
- `feature/xxx`：新功能开发分支。
- `fix/xxx`：普通缺陷修复分支。
- `hotfix/xxx`：线上或交付版本紧急修复分支，通常从 `main` 拉出。
- `release/xxx`：发布准备分支，用于版本冻结、测试、修复发布前问题。
- `refactor/xxx`：重构分支，不改变外部行为。
- `docs/xxx`：文档修改分支。

基本原则：

- 禁止直接向 `main` 提交代码。
- 禁止直接在 `main` 上开发。
- 所有开发都必须基于独立分支完成。
- 所有合并都应通过 Pull Request。
- `main` 和 `dev` 应开启 GitHub Branch Protection。
- 合并前必须通过测试、Review 和必要的文档检查。

### 1.2 分支职责

| 分支 | 来源 | 合并目标 | 职责 |
| --- | --- | --- | --- |
| `main` | `release/*` 或 `hotfix/*` | 无 | 稳定版本、课程提交版、展示版 |
| `dev` | `main` 或长期维护 | `release/*` / `main` | 团队日常集成 |
| `feature/xxx` | `dev` | `dev` | 新功能开发 |
| `fix/xxx` | `dev` | `dev` | 普通 bug 修复 |
| `hotfix/xxx` | `main` | `main` 和 `dev` | 紧急修复稳定版本 |
| `release/xxx` | `dev` | `main` 和 `dev` | 发布前测试和冻结 |
| `refactor/xxx` | `dev` | `dev` | 代码结构优化 |
| `docs/xxx` | `dev` | `dev` | 文档维护 |

### 1.3 分支命名规范

命名格式：

```bash
<type>/<short-description>
```

推荐示例：

```bash
feature/data-loader
feature/model-training-pipeline
fix/readme-typo
fix/dataset-path-error
hotfix/missing-config
refactor/training-loop
docs/git-workflow
release/v1.0.0
```

注意事项：

- 使用英文小写。
- 单词之间使用 `-`。
- 名称表达任务内容，不要写成 `feature/test`、`fix/bug`。
- 一个分支只做一类任务。
- 不要在一个分支里混合功能开发、格式化、重构和文档大改。

### 1.4 多人协作推荐流程

1. 在 GitHub Issue 中登记任务。
2. 从 `dev` 拉出自己的分支。
3. 本地开发并保持小步提交。
4. 每天开始开发前同步远程 `dev`。
5. 发起 PR 前再次同步 `dev`。
6. 通过测试后提交 Pull Request。
7. 至少一名成员 Review。
8. 修复 Review 意见。
9. 合并到 `dev`。
10. 删除远程和本地功能分支。
11. 发布时从 `dev` 拉出 `release/vx.y.z`，测试通过后合并到 `main` 并打 tag。

## 2. Git 操作指令

### 2.1 克隆项目

命令：

```bash
git clone <repo-url>
cd <repo-name>
```

作用解释：

- 从 GitHub 下载项目到本地。
- 自动配置默认远程仓库 `origin`。

使用场景：

- 第一次参与项目。
- 换电脑或重新初始化开发环境。

常见错误与注意事项：

- 如果没有权限，检查是否加入 GitHub 仓库协作者。
- HTTPS 方式可能需要 Personal Access Token。
- SSH 方式需要提前配置 SSH Key。

### 2.2 查看与配置远程仓库

命令：

```bash
git remote -v
git remote add origin <repo-url>
git remote set-url origin <new-repo-url>
```

作用解释：

- `git remote -v` 查看远程仓库地址。
- `git remote add origin` 添加远程仓库。
- `git remote set-url origin` 修改远程仓库地址。

使用场景：

- 本地仓库还没有关联 GitHub。
- 仓库地址从 HTTPS 切换到 SSH。
- 项目迁移到新仓库。

常见错误与注意事项：

- 如果提示 `remote origin already exists`，使用 `git remote set-url origin <url>`。
- 推送失败时先检查远程地址是否正确。

### 2.3 拉取最新代码

命令：

```bash
git checkout dev
git pull origin dev
```

作用解释：

- 切换到 `dev`。
- 从远程同步最新集成代码。

使用场景：

- 每天开始开发前。
- 创建新分支前。
- 发起 PR 前。

常见错误与注意事项：

- 如果本地有未提交修改，`checkout` 或 `pull` 可能失败。
- 可先使用 `git status` 检查工作区。
- 不确定是否保留本地修改时，先 `git stash`。

### 2.4 创建并切换开发分支

命令：

```bash
git checkout dev
git pull origin dev
git checkout -b feature/data-loader
```

或使用新版命令：

```bash
git switch dev
git pull origin dev
git switch -c feature/data-loader
```

作用解释：

- 基于最新 `dev` 创建功能分支。

使用场景：

- 开始新功能、新修复、新文档任务。

常见错误与注意事项：

- 不要从过期的本地 `dev` 创建分支。
- 不要从自己的旧 feature 分支继续开新任务。
- 分支名要能对应具体 Issue 或任务。

### 2.5 查看当前修改

命令：

```bash
git status
git diff
git diff --staged
```

作用解释：

- `git status` 查看哪些文件被修改。
- `git diff` 查看未暂存修改。
- `git diff --staged` 查看已暂存修改。

使用场景：

- 提交代码前检查变更。
- Review 自己是否误改了无关文件。

常见错误与注意事项：

- 提交前必须看 `git diff`。
- 避免把数据集、临时文件、IDE 配置误提交。
- 如果误修改无关文件，不要随手提交。

### 2.6 暂存并提交代码

命令：

```bash
git add <file>
git commit -m "feat: add stock data loader"
```

或暂存所有已跟踪和新增文件：

```bash
git add .
git commit -m "fix: correct dataset path handling"
```

作用解释：

- `git add` 把修改放入暂存区。
- `git commit` 生成一次本地提交。

使用场景：

- 完成一个小功能。
- 修复一个明确问题。
- 补充一段文档。

常见错误与注意事项：

- 不要无脑 `git add .`，先用 `git status` 和 `git diff` 检查。
- 不要提交大数据文件、模型权重、缓存目录。
- 一个 commit 只表达一个清晰变更。

### 2.7 推送远程分支

命令：

```bash
git push -u origin feature/data-loader
```

后续推送同一分支：

```bash
git push
```

作用解释：

- 将本地分支推送到 GitHub。
- `-u` 设置上游分支，后续可直接 `git push`。

使用场景：

- 首次推送功能分支。
- 想让队友查看代码。
- 准备创建 Pull Request。

常见错误与注意事项：

- 如果提示无权限，检查仓库权限和 GitHub 认证。
- 分支推错时不要强行覆盖远程分支，先确认当前分支。

### 2.8 发起 PR 前同步 `dev`

推荐方式一：merge 同步。

```bash
git checkout feature/data-loader
git fetch origin
git merge origin/dev
```

推荐方式二：rebase 同步。

```bash
git checkout feature/data-loader
git fetch origin
git rebase origin/dev
```

作用解释：

- 把 `dev` 的最新代码同步到自己的功能分支。
- 提前发现冲突，减少 PR 合并风险。

使用场景：

- 发起 PR 前。
- 分支开发时间超过一天。
- 队友修改了相同模块。

常见错误与注意事项：

- 新手优先使用 `merge`，历史更直观。
- 熟悉 Git 后可使用 `rebase` 保持提交历史线性。
- 已经推送给多人共同使用的分支，谨慎 rebase。

### 2.9 解决 merge conflict

触发冲突示例：

```bash
git merge origin/dev
```

查看冲突文件：

```bash
git status
```

解决步骤：

1. 打开冲突文件。
2. 搜索冲突标记：

```text
<<<<<<< HEAD
当前分支内容
=======
被合并分支内容
>>>>>>> origin/dev
```

3. 人工决定保留哪部分或如何合并。
4. 删除冲突标记。
5. 重新暂存并提交。

命令：

```bash
git add <conflict-file>
git commit
```

如果是 rebase 冲突：

```bash
git add <conflict-file>
git rebase --continue
```

取消 merge：

```bash
git merge --abort
```

取消 rebase：

```bash
git rebase --abort
```

作用解释：

- 手动协调两个分支对同一位置的修改。

使用场景：

- 多人修改同一个文件。
- 一个分支重命名文件，另一个分支修改文件内容。

常见错误与注意事项：

- 不要直接删除队友代码。
- 冲突解决后必须运行测试或至少执行相关脚本。
- 不确定业务含义时，联系对应作者确认。

### 2.10 创建 Pull Request

命令行推送后，在 GitHub 页面创建 PR：

```bash
git push -u origin feature/data-loader
```

PR 方向：

```text
feature/data-loader -> dev
```

作用解释：

- 让团队成员 Review 代码。
- 通过 GitHub 检查、讨论、合并变更。

使用场景：

- 功能完成。
- 修复完成。
- 文档修改完成。

常见错误与注意事项：

- 不要把 feature 分支直接 PR 到 `main`。
- PR 描述要说明做了什么、如何测试、关联哪个 Issue。

### 2.11 合并后删除分支

删除远程分支：

```bash
git push origin --delete feature/data-loader
```

删除本地分支：

```bash
git checkout dev
git pull origin dev
git branch -d feature/data-loader
```

强制删除本地分支：

```bash
git branch -D feature/data-loader
```

作用解释：

- 清理已经完成的任务分支。

使用场景：

- PR 已合并。
- 分支不再需要。

常见错误与注意事项：

- 未合并分支不要随便 `-D`。
- 删除前确认 PR 已经合并。

### 2.12 tag 版本发布

命令：

```bash
git checkout main
git pull origin main
git tag -a v1.0.0 -m "release: v1.0.0"
git push origin v1.0.0
```

查看 tag：

```bash
git tag
git show v1.0.0
```

作用解释：

- 给稳定版本打不可变版本标记。

使用场景：

- 课程阶段提交。
- 实验复现实验结果。
- 发布可展示版本。

常见错误与注意事项：

- tag 应打在 `main` 的稳定提交上。
- 不要给未测试代码打正式版本 tag。
- 如果 tag 打错，应先沟通再删除，避免影响他人。

删除本地 tag：

```bash
git tag -d v1.0.0
```

删除远程 tag：

```bash
git push origin --delete v1.0.0
```

### 2.13 回滚错误提交

推荐方式：生成反向提交。

```bash
git revert <commit-hash>
git push
```

作用解释：

- 保留历史记录，并用新提交撤销错误提交。

使用场景：

- 错误代码已经推送到远程。
- 错误提交已经被他人同步。
- `main` 或 `dev` 需要安全回滚。

常见错误与注意事项：

- 公共分支优先使用 `git revert`。
- 不要在公共分支随意使用 `git reset --hard`。

仅本地撤销最近一次提交但保留修改：

```bash
git reset --soft HEAD~1
```

仅本地撤销最近一次提交并取消暂存：

```bash
git reset --mixed HEAD~1
```

危险操作：

```bash
git reset --hard HEAD~1
```

注意：

- `--hard` 会丢弃工作区修改。
- 对公共分支使用前必须确认没有影响他人。

### 2.14 stash 临时保存修改

命令：

```bash
git stash
git stash list
git stash pop
```

带说明保存：

```bash
git stash push -m "wip: temporary data preprocessing changes"
```

应用但不删除 stash：

```bash
git stash apply stash@{0}
```

删除 stash：

```bash
git stash drop stash@{0}
```

作用解释：

- 临时保存未提交修改，让工作区恢复干净。

使用场景：

- 需要临时切换分支。
- pull 前本地有未提交修改。
- 当前工作还不适合提交。

常见错误与注意事项：

- `stash pop` 可能产生冲突。
- stash 不适合作为长期保存方案。
- 重要修改应尽快提交到分支。

### 2.15 查看日志与 diff

命令：

```bash
git log --oneline --graph --decorate --all
git log --stat
git show <commit-hash>
git diff dev...feature/data-loader
git diff --name-only dev...feature/data-loader
```

作用解释：

- 查看提交历史、分支关系和具体代码差异。

使用场景：

- Review 前自查。
- 查找某次改动来源。
- 对比 feature 分支与 `dev` 的差异。

常见错误与注意事项：

- 发 PR 前建议查看 `git diff --name-only dev...当前分支`。
- 如果出现大量无关文件，说明分支污染，需要拆分或清理。

## 3. Commit 规范

### 3.1 Conventional Commits 格式

标准格式：

```text
<type>(optional scope): <description>
```

常用类型：

| 类型 | 含义 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | bug 修复 |
| `docs` | 文档变更 |
| `refactor` | 重构，不改变外部行为 |
| `test` | 测试相关 |
| `chore` | 构建、配置、依赖、杂项 |
| `perf` | 性能优化 |
| `style` | 代码格式，不改变逻辑 |

### 3.2 优秀 commit message 示例

```text
feat(data): add A-share CSV loader
fix(train): handle missing labels in dataset
docs(git): add collaboration workflow
refactor(model): split trainer from evaluation logic
test(data): add tests for date range filtering
chore: update gitignore for local datasets
perf(loader): cache parsed trading calendar
style: format preprocessing scripts
```

### 3.3 错误示例

```text
update
fix bug
add files
change code
final version
temporary commit
my work
```

问题：

- 不知道改了什么。
- 不知道为什么改。
- 无法快速定位历史。
- Review 和回滚成本高。

### 3.4 保持 commit 粒度清晰

建议：

- 一个 commit 只解决一个问题。
- 能独立描述的修改应拆成不同 commit。
- 不要把格式化和功能修改混在一起。
- 不要把数据文件、模型文件和代码逻辑混在一起。
- 提交前检查 `git diff --staged`。

推荐粒度：

```text
feat(data): add dataset loading function
test(data): cover invalid file path handling
docs(data): document dataset directory layout
```

不推荐粒度：

```text
feat: add loader, train model, update docs, fix chart, format project
```

## 4. 协作规范

### 4.1 Pull Request 规范

PR 标题格式：

```text
feat(data): add A-share dataset loader
```

PR 描述建议包含：

```markdown
## Summary
- Added CSV loader for A-share dataset
- Normalized date and ticker columns

## Test
- [ ] Ran unit tests
- [ ] Ran training smoke test
- [ ] Checked docs or examples

## Related Issue
Closes #12

## Notes
Known limitations or follow-up work.
```

PR 要求：

- 一个 PR 只解决一个明确任务。
- PR 尽量小而清晰。
- 大 PR 必须拆分阶段或写清楚 Review 顺序。
- 合并前确保分支已同步最新 `dev`。
- PR 中不要包含无关格式化和临时文件。

### 4.2 Code Review 规范

Reviewer 关注点：

- 逻辑是否正确。
- 是否有边界条件遗漏。
- 是否破坏已有功能。
- 是否引入不必要复杂度。
- 是否需要测试或文档。
- 是否存在数据路径、随机种子、环境依赖等复现问题。

Author 关注点：

- PR 描述清晰。
- 提前自查 diff。
- 对 Review 意见及时回应。
- 修改后说明做了哪些调整。

Review 交流原则：

- 针对代码和问题，不针对个人。
- 对阻塞问题明确标注。
- 对建议类问题说明原因。
- 意见不一致时，以项目目标、可维护性和可复现性为准。

### 4.3 Issue 管理规范

Issue 类型：

- `feature`：新功能。
- `bug`：错误修复。
- `docs`：文档任务。
- `experiment`：实验设计或结果记录。
- `refactor`：重构任务。
- `question`：待讨论问题。

Issue 内容建议：

```markdown
## Background
为什么需要做这个任务。

## Task
- [ ] 子任务 1
- [ ] 子任务 2

## Expected Result
完成后应该达到什么效果。

## Related Files
可能涉及哪些文件或模块。
```

使用建议：

- 每个开发分支最好对应一个 Issue。
- PR 中使用 `Closes #issue-number` 自动关联。
- 大任务拆成多个可完成的小 Issue。

### 4.4 冲突解决原则

原则：

- 先理解双方修改目的，再解决冲突。
- 不确定时联系对应作者。
- 保留正确逻辑，而不是简单选择自己的版本。
- 解决后必须运行相关测试。
- 对公共接口、数据格式、配置文件的冲突要格外谨慎。

高风险冲突文件：

- 数据处理入口文件。
- 模型结构文件。
- 训练配置文件。
- README 和实验说明。
- 依赖配置文件。

### 4.5 文件修改原则

建议：

- 修改前先确认自己负责的模块范围。
- 不在同一 PR 中修改大量无关文件。
- 不提交本地数据集、缓存、日志、模型权重。
- 大文件使用 Git LFS 或外部存储。
- 配置模板可提交，个人本地配置不提交。

推荐：

```text
config.example.yaml
.env.example
```

不推荐提交：

```text
.env
local_config.yaml
*.pth
*.ckpt
*.log
A股数据/
```

### 4.6 大功能开发建议

大功能不要一次性提交巨大 PR，推荐拆分：

1. 数据结构或接口定义。
2. 核心逻辑实现。
3. 测试与样例。
4. 文档和使用说明。
5. 性能优化或重构。

长期分支维护建议：

- 每天同步一次 `dev`。
- 每完成一个阶段就开 PR。
- 不要让 feature 分支长期落后主线。
- 对接口变更提前在团队中沟通。

### 4.7 多人同时开发时避免覆盖代码的方法

建议：

- 任务拆分到不同模块或文件。
- 开发前在 Issue 中声明负责人。
- 每天开发前 `git pull origin dev`。
- 发 PR 前同步 `dev`。
- 不直接修改他人负责的大段代码。
- 对公共函数、配置结构、数据格式的修改提前沟通。
- 使用小 PR 降低冲突范围。
- Review 时重点看是否误删他人逻辑。

## 5. 版本管理

### 5.1 SemVer 版本号规范

使用语义化版本：

```text
MAJOR.MINOR.PATCH
```

示例：

```text
v1.0.0
v1.1.0
v1.1.1
```

含义：

- `MAJOR`：不兼容变更。
- `MINOR`：向后兼容的新功能。
- `PATCH`：向后兼容的问题修复。

### 5.2 版本示例区别

`v1.0.0`：

- 第一个稳定版本。
- 可用于课程阶段提交、实验复现、公开展示。

`v1.1.0`：

- 在 `v1.0.0` 基础上新增功能。
- 例如新增模型、数据处理流程或实验配置。
- 不破坏原有使用方式。

`v1.1.1`：

- 修复 `v1.1.0` 的 bug。
- 例如修复路径错误、文档错误、边界条件错误。
- 不新增大型功能。

### 5.3 release tag 规范

tag 命名：

```text
v<MAJOR>.<MINOR>.<PATCH>
```

示例：

```text
v1.0.0
v1.1.0
v1.1.1
```

发布分支命名：

```text
release/v1.0.0
release/v1.1.0
```

tag message：

```bash
git tag -a v1.0.0 -m "release: v1.0.0"
```

### 5.4 Changelog 维护方式

建议在根目录维护：

```text
CHANGELOG.md
```

推荐格式：

```markdown
# Changelog

## [v1.1.0] - 2026-05-24

### Added
- Added model training pipeline.

### Fixed
- Fixed dataset path handling on Windows.

### Changed
- Refactored data preprocessing interface.

## [v1.0.0] - 2026-05-20

### Added
- Initial stable project version.
```

分类建议：

- `Added`：新增功能。
- `Fixed`：问题修复。
- `Changed`：行为变化。
- `Removed`：删除内容。
- `Docs`：文档变化。
- `Security`：安全修复。

### 5.5 GitHub Releases 发布版本

推荐流程：

1. 确保 `dev` 已完成测试。
2. 创建发布分支：

```bash
git checkout dev
git pull origin dev
git checkout -b release/v1.0.0
```

3. 只允许修复发布相关问题，不再加入新功能。
4. 更新 `CHANGELOG.md` 和版本号。
5. 提交发布准备：

```bash
git add CHANGELOG.md
git commit -m "chore(release): prepare v1.0.0"
git push -u origin release/v1.0.0
```

6. 创建 PR：`release/v1.0.0 -> main`。
7. 合并后在 `main` 打 tag：

```bash
git checkout main
git pull origin main
git tag -a v1.0.0 -m "release: v1.0.0"
git push origin v1.0.0
```

8. 将发布修复同步回 `dev`：

```bash
git checkout dev
git pull origin dev
git merge main
git push origin dev
```

9. 在 GitHub 页面进入 `Releases`。
10. 点击 `Draft a new release`。
11. 选择 tag `v1.0.0`。
12. 填写标题和 changelog。
13. 上传必要附件，例如报告、模型说明、复现实验说明。
14. 发布 Release。

## 6. 推荐工作流：从开发到合并

### 6.1 新功能开发完整流程

```bash
git clone <repo-url>
cd <repo-name>

git checkout dev
git pull origin dev

git checkout -b feature/data-loader

# 开发代码
git status
git diff

git add <changed-files>
git commit -m "feat(data): add A-share dataset loader"

git fetch origin
git merge origin/dev

# 如有冲突，解决冲突后：
git add <conflict-files>
git commit

# 运行测试或检查

git push -u origin feature/data-loader
```

然后在 GitHub 上创建 PR：

```text
feature/data-loader -> dev
```

PR 合并后：

```bash
git checkout dev
git pull origin dev
git branch -d feature/data-loader
git push origin --delete feature/data-loader
```

### 6.2 紧急修复流程

```bash
git checkout main
git pull origin main
git checkout -b hotfix/missing-config

# 修复问题
git add <changed-files>
git commit -m "fix(config): restore missing default config"
git push -u origin hotfix/missing-config
```

在 GitHub 上创建 PR：

```text
hotfix/missing-config -> main
```

合并后同步回 `dev`：

```bash
git checkout dev
git pull origin dev
git merge main
git push origin dev
```

必要时发布补丁版本：

```bash
git checkout main
git pull origin main
git tag -a v1.0.1 -m "release: v1.0.1"
git push origin v1.0.1
```

## 7. 日常开发命令速查表

| 场景 | 命令 |
| --- | --- |
| 查看状态 | `git status` |
| 查看修改 | `git diff` |
| 查看已暂存修改 | `git diff --staged` |
| 查看分支 | `git branch` |
| 查看所有分支 | `git branch -a` |
| 切换分支 | `git checkout dev` |
| 创建分支 | `git checkout -b feature/xxx` |
| 拉取最新代码 | `git pull origin dev` |
| 暂存文件 | `git add <file>` |
| 提交代码 | `git commit -m "feat: xxx"` |
| 推送分支 | `git push -u origin feature/xxx` |
| 后续推送 | `git push` |
| 拉取远程信息 | `git fetch origin` |
| 合并 dev | `git merge origin/dev` |
| rebase dev | `git rebase origin/dev` |
| 中止 merge | `git merge --abort` |
| 中止 rebase | `git rebase --abort` |
| 查看历史 | `git log --oneline --graph --decorate --all` |
| 查看某次提交 | `git show <commit-hash>` |
| 临时保存修改 | `git stash push -m "wip: message"` |
| 查看 stash | `git stash list` |
| 恢复 stash | `git stash pop` |
| 删除本地分支 | `git branch -d feature/xxx` |
| 删除远程分支 | `git push origin --delete feature/xxx` |
| 创建 tag | `git tag -a v1.0.0 -m "release: v1.0.0"` |
| 推送 tag | `git push origin v1.0.0` |
| 安全回滚提交 | `git revert <commit-hash>` |

## 8. 最佳实践总结

- 每天开始开发前先同步 `dev`。
- 所有开发都在独立分支完成。
- 一个分支只做一个任务。
- 一个 commit 只表达一个清晰修改。
- 提交前必须看 `git status` 和 `git diff`。
- PR 前同步 `dev`，提前解决冲突。
- 通过 PR 合并，不直接推送 `main`。
- 数据集、缓存、日志、模型权重不要提交到 Git。
- 大功能拆成多个小 PR。
- 修改公共接口前先沟通。
- 冲突解决后必须测试。
- 发布版本必须从稳定的 `main` 打 tag。
- 重要版本维护 `CHANGELOG.md` 和 GitHub Release。

## 9. 常见错误案例

### 9.1 直接在 main 上开发

错误：

```bash
git checkout main
# 修改代码
git add .
git commit -m "update"
git push origin main
```

问题：

- 破坏稳定分支。
- 绕过 Review。
- 容易把未测试代码带入发布版本。

正确做法：

```bash
git checkout dev
git pull origin dev
git checkout -b feature/xxx
```

### 9.2 提交信息过于模糊

错误：

```bash
git commit -m "fix"
git commit -m "update code"
```

正确：

```bash
git commit -m "fix(data): handle missing trading date"
git commit -m "feat(model): add baseline LSTM model"
```

### 9.3 一个 PR 修改太多内容

错误：

```text
同时修改数据处理、模型结构、训练脚本、README、依赖文件和格式化全仓库。
```

问题：

- Review 成本高。
- 冲突概率高。
- 出错后难以回滚。

正确做法：

- 拆成多个 PR。
- 先合并基础接口，再合并模型实现，再补测试和文档。

### 9.4 提交了本地数据或临时文件

错误：

```text
A股数据/
*.log
__pycache__/
.env
```

正确做法：

- 在 `.gitignore` 中忽略这些文件。
- 提交前用 `git status` 检查。

如果已经提交到本地但还没 push：

```bash
git rm --cached <file>
git commit -m "chore: remove local generated files"
```

### 9.5 未同步 dev 就发起 PR

问题：

- 合并时才发现冲突。
- CI 或测试失败。
- 影响其他人合并节奏。

正确做法：

```bash
git fetch origin
git merge origin/dev
```

或：

```bash
git fetch origin
git rebase origin/dev
```

### 9.6 用 reset 修改公共历史

危险操作：

```bash
git reset --hard HEAD~1
git push --force
```

问题：

- 可能删除队友已经基于它开发的提交。
- 导致团队历史不一致。

正确做法：

```bash
git revert <commit-hash>
git push
```

### 9.7 解决冲突时误删队友代码

错误做法：

- 看到冲突后直接保留自己的版本。
- 删除冲突标记但不理解逻辑。
- 解决后不运行测试。

正确做法：

- 阅读双方修改。
- 必要时联系作者。
- 合并两边必要逻辑。
- 运行相关测试。

## 10. GitHub 仓库建议配置

建议开启 Branch Protection：

- `main` 禁止直接 push。
- `main` 合并前必须经过 PR。
- `main` 合并前至少 1 人 Review。
- `main` 合并前必须通过测试。
- `dev` 建议也通过 PR 合并。

建议使用 GitHub 标签：

- `feature`
- `bug`
- `docs`
- `experiment`
- `refactor`
- `priority-high`
- `good first issue`

建议使用 GitHub 项目看板：

- `Todo`
- `In Progress`
- `Review`
- `Done`

