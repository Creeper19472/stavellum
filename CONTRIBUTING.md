# 贡献与 Pull Request 规范

Stavellum 当前面向 Windows x64，使用 Python 3.14、PySide6 6.11.2 与 Rust。
提交前先描述可复现的问题和预期行为，再选择最小、完整的修改范围。

## PR 标题和范围

标题使用 `type(scope): 描述`，scope 可省略，描述可以用中文或英文。例如：

- `fix(gui): 重新导入后保留工程保存路径`
- `feat(ui): 统一桌面主题和创作工具栏`
- `docs: 补充贡献与验证规范`

type 为 `feat`、`fix`、`refactor`、`perf`、`docs`、`test`、`build`、`ci` 或 `chore`。
不兼容变更在冒号前加 `!`，并说明迁移办法。
一次 PR 解决一个明确问题；配套文档、回归测试和直接相关修复可一起提交。
大规模迁移按可验证的功能边界拆分，尚未具备功能对等性时保留现有路径。
推荐 squash 合并，合并提交沿用规范标题；合并方式和必需检查由仓库维护者配置。

## PR 描述

使用 [.github/pull_request_template.md](.github/pull_request_template.md)，填写：

1. **问题与结果**：触发条件、修改前后的行为；关联 issue 时使用 `Closes #编号`。
2. **变更范围**：最终实现、兼容性与必要的取舍，避免罗列开发过程。
3. **验证证据**：实际命令、通过/失败/跳过数量和跳过原因；不要沿用其他提交的测试数字。
4. **UI 或性能证据**：UI 附实际运行截图、窗口尺寸及缩放比例；性能附硬件、输入、分辨率、帧率、后端、重复次数和可复现命令。
5. **限制与迁移**：未验证的设备、格式或路径，以及确实需要的后续工作。

更新 PR 范围后同步修改标题和描述。历史说明 [docs/PR_DESCRIPTION.md](docs/PR_DESCRIPTION.md)
属于 Rust 迁移的记录，不是通用模板，也不能作为新提交的验证结论。

## 本地检查

```powershell
uv sync --extra dev --locked
uv run ruff check src tests scripts
uv run pytest -q -m "not integration and not slow"
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

默认检查通过后，无需因低影响的文字或样式修改重复运行昂贵测试。
修复行为问题应补能复现问题的回归测试，避免仅复述实现。
本机默认 pytest 临时目录不可写时，可使用新的、专用于本次测试的
`--basetemp artifacts/pytest-<本次运行唯一名称>`。pytest 会清空该目录，勿指向已有作品。

涉及原生 ABI、渲染、音频、编码或导出时，先重建 DLL 再测试；忽略的本机 DLL
可能早于源码，不能代表当前提交。需要有效 Vulkan 设备及 FFmpeg/FFprobe：

```powershell
uv run python scripts/build_rust.py --install
uv run pytest -q
```

Qt C++ 后端变更另按 [docs/rust-renderer.md](docs/rust-renderer.md) 构建对应后端。
涉及打包时验证 `uv build --wheel` 并检查实际包资源。
新增或更新依赖需同步锁文件与适用的第三方许可。

UI 截图可用 `uv run python scripts/preview_ui.py --output artifacts/ui-preview` 生成。
脚本使用独立设置与实际示例谱面，包含启动封面、欢迎页空态/最近工程、编辑器各页、较小窗口和新建向导。
公开 PR 截图加 `--public-paths`，将示例中的本机路径替换为公开占位路径。
设置 `QT_SCALE_FACTOR=1.25` 或 `1.5` 后再次运行，可检查不同缩放比例。

## Review 与合并条件

- 先检查数据丢失、崩溃、错误输出、资源/进程泄漏，再检查性能与可维护性。
- 对问题注明触发条件、影响、具体位置及建议；P0 为紧急阻断，P1 为高优先级，P2 为常规修复，P3 为低影响建议。
- 工程格式、CLI、`sprhi_*`/`spcore_*` ABI、颜色/像素布局与 CPU 回退变化必须明确说明，并提供相应验证。
- 导入失败/取消时保留现有工程；保存失败时保留原文件与未保存状态；取消、关闭和迟到结果不可破坏界面状态。
- UI 检查最小支持尺寸、键盘导航、长文件名、空态、禁用态、忙碌态，以及常用 Windows 缩放比例。
- 所需 CI 检查通过、阻断问题已解决、描述与当前 diff 一致后再合并。设备相关验证的跳过应明确记录。

仓库内的 CI 检查 PR 标题、Python 静态检查和常规测试，以及 Rust 格式、单元测试与 clippy。
CI 不提供 GPU 导出验证；维护者应在 GitHub 分支保护中将需要的检查设为必需。
工作流使用官方 [checkout](https://github.com/actions/checkout) 和
[setup-python](https://github.com/actions/setup-python) actions。
