<p align="center">
  <img src="src/stavellum/assets/logo.png" alt="Stavellum logo" width="144" />
</p>
<h1 align="center">Stavellum</h1>

<p align="center"> <b>简体中文</b> | <a href="./README.md">English</a> </p>

基于受支持格式的自动乐谱化与演示实用程序。

将 FL Studio 工程或 MIDI 转换为五线谱表，并使其在画面上滚动，以制作精美的演示视频。导入 **FLP / MIDI + 原曲音频**，校正乐器与记谱设置，预览并导出 **MP4、MusicXML、PDF 分谱**。

本程序采用逐帧渲染而非屏幕截取的实现方案，默认输出 1920×1080、60 fps、H.264 + AAC，优先使用 **RHI Vulkan** 帧合成及 NVIDIA NVENC 编码，设备不可用时自动回退到 CPU。

## 安装与启动

为使本程序在性能上取得最佳表现，设备应至少具备**支持 Vulkan 与 4×MSAA 的显卡及其驱动**。

目前为止，程序仅支持在 Windows x64 上运行，需要 Python **3.14**、[uv](https://docs.astral.sh/uv/) 和提供 `libx264` 编码器的 FFmpeg。源码运行还需要 Cargo，以及 Visual Studio C++ 开发工具或 windows-gnu Rust 工具链，以构建原生 Rust Vulkan 后端（见 [Rust 构建说明](docs/rust-renderer.md)）。可选的传统 Qt RHI 后端另外需要 Visual Studio C++ 开发工具，并要求 PySide6 **6.11.2** 与其 Qt 私有 ABI 保持一致。

将 `ffmpeg.exe`、`ffprobe.exe` 所在目录加入 `PATH`，然后在项目目录运行：

```powershell
uv python install 3.14
uv sync --extra dev --locked
uv run python scripts/build_rust.py --install
ffmpeg -version
ffprobe -version
.\run.ps1
```

也可以运行 `uv run stavellum gui`，或用 `uv run python -m stavellum gui`。打开已有项目：

```powershell
.\run.ps1 -Project "artifacts/demo/demo.stproj"
```

可先试用由程序生成的示例：

```powershell
uv run stavellum demo --output artifacts/demo --bars 16 --open
```

这将生成示例工程对应的 `.stproj`、MIDI 和合成 WAV，包含普通／Pizz 奏法、长音、三连音与静默段。可利用该示例工程检验程序行为。

## 快速上手

### 新建与导入
1.  从编辑器界面顶端的“文件”菜单或欢迎界面点击**新建工程**，选择 FLP 或 MIDI 文件，并关联原曲音频（也可后续补选）。
2.  在弹出的向导中核对曲目信息、 Arrangement、速度与拍号；确认后点击“重新导入”。

*注：打开保存的 `.stproj` 项目文件可直接恢复上次的工作状态。*

### 确认速度与拍号
若 FLP 文件包含未识别的自动化，系统会提示人工确认曲目整体的时钟与速度。

### 声部与分谱管理
* **合并与拆分**：可多选分谱进行合并（如将普通、Pizz、Staccato 奏法合并），或按来源/ MIDI 通道重新拆分。
* **排版设置**：为各声部配置乐器图标、谱号、调号、移调、量化及三连音，或开启双谱表与打击乐映射。

### 记谱优化（可选）
* **自动简化音符记法**（默认开启）：优先使用调号内的等音写法（如 E♯、B♯ 等），大幅减少多余的临时还原号。
* **自动八度移位**（默认关闭）：开启后，连续高/低音区将自动使用 **8va、8vb、15ma、15mb** 标记，减少加线，使谱面更易读。

### 预览与对齐
* 点击顶部“更新预览”，通过下方播放轴核对音画同步。
* 若音频与乐谱未对齐，可在“来源与时间”中调整**音频内乐谱零点**（如输入 `1.25` 表示音频第 1.25 秒对应乐谱第一拍）。

### 导出作品
* 保存工程文件（`.stproj`）后，通过“导出”菜单导出 **MP4 视频** 或 **MusicXML / PDF 分谱**。

## 核心界面与参数配置

通过“设置”面板可对画面与视觉动画进行个性化配置：

* **曲目与文字信息**：设置曲名、副标题、署名及报幕内容，可自由调整位置与字号。
* **画面与谱区布局**：自定视频尺寸与谱区边距；声部较少时，画面会自动适度放大以优化观感。
* **开场与报幕**：
  * 可设置**开场停顿**时长，实现“静态展示报幕/画面 $\rightarrow$ 淡出后平滑展开谱区 $\rightarrow$ 启动播放”的开场效果。
  * 可配置报幕文字的定时淡出。


* **声部动画与流畅度**：
  * **动画速度**：预设提供“快速、中速、慢速、极慢”及自定义选项，控制声部进退场与布局平滑过渡的时长。
  * **动画后最短稳定时间**（默认 2 秒）：避免声部在短时间内频繁隐藏、出现或闪烁缩放；设为 `0` 可关闭此限制。


* **渲染与导出**：视频导出在后台独立进程中进行，支持随时查看进度或取消。若取消进度窗口，可从主窗口底部的“导出详情”重新打开。

> [!NOTE]
> 项目文件（`.stproj`）仅保存设置与输入文件的引用路径。若移动或重命名了原曲音频，重新打开工程时需要手动补选音频。

## 可选功能

### 自定义乐器图标

在“分谱校正”的图标设置旁单击 **选择图标**，可以选择项目内置图标、导入 **SVG / PNG / JPEG / WebP**，或在 Font Awesome 页按英文名称搜索（例如 `guitar`、`music`），按来源和样式过滤。

> [!NOTE]
> SVG 必须自包含；引用外部图片时请先在制图软件中将图片嵌入 SVG。另外，静态图标不会播放 SVG 动画。

应用随包提供 **Font Awesome Free 7.3.1** 的 Solid、Regular、Brands，共 2,883 个图标，离线可用。也可以点 **连接 Free / Pro 图标库**，选择自己已解压的 Font Awesome 发行包根目录、`svgs-full` 或 `svgs`。应用仅在选择图标时才会读取其内容，而不会将整个图标库复制进项目。

也可直接输入原有图标名称（如 `violin`）或 `fa:solid:guitar`、`fa:regular:bell` 等内置 Free 名称。留空表示自动按乐器匹配；**恢复自动**清空已有选择；取消“使用乐器图标”即可隐藏图标，图标设置暂时禁用但保留选择，再次勾选便可恢复。改变乐器不会覆盖已经明确选择的图标。

### 展示软件 Logo

可在设置页中开启默认关闭的“在画面右下角显示软件 Logo”功能。启用后，可以选择“常驻”“淡入后常驻”或“开场淡入，停留后淡出”等显示模式，并调整徽标的大小、最大不透明度及淡入／停留／淡出时长等参数。

如果您喜爱本项目，不妨利用该功能表示对项目的支持。:)

## 命令行

`inspect` / `import` 的 Arrangement 索引从 **0** 开始。下面的输入名称是示例，请替换为自己的文件：

```powershell
# 只读检查来源、音符数量和诊断
uv run stavellum inspect "song.flp" --arrangement 0

# 创建可在桌面校正的项目；仅在确已核实固定时钟后加确认参数
uv run stavellum import "song.flp" --audio "song.flac" --output "song.stproj" --confirm-fixed-timing

# MIDI 后备入口
uv run stavellum import "song.mid" --audio "song.wav" --output "song.stproj"

# 打开项目、输出指定音频时间的 PNG 帧
uv run stavellum gui "song.stproj"
uv run stavellum frame "song.stproj" --time 12.5 --output artifacts/frame.png

# 离线逐帧导出；尺寸/帧率覆盖只影响本次导出
uv run stavellum render "song.stproj" --output artifacts/video.mp4 --width 1920 --height 1080 --fps 60 --preset medium

# 明确选择 RHI Vulkan 合成与 NVENC；不可用时直接报错
uv run stavellum render "song.stproj" --output artifacts/gpu.mp4 --render-backend gpu --video-encoder h264_nvenc --nvenc-cq 18 --nvenc-preset p5

# 固定使用 CPU，或仅对本次命令关闭自动识别
uv run stavellum render "song.stproj" --output artifacts/cpu.mp4 --render-backend cpu --video-encoder libx264 --no-auto-staccato --no-auto-grace --no-auto-dynamics --no-auto-ottava

# 每个已启用乐器生成 MusicXML 与分页 PDF
uv run stavellum parts "song.stproj" --output artifacts/parts
```

`import`、`frame`、`render`、`parts` 均支持 `--auto-simplify-accidentals`／`--no-auto-simplify-accidentals`，统一开启／关闭所有分谱的记法简化；不指定时沿用项目中保存的逐分谱选择，新导入默认开启。`frame`、`render`、`parts` 的覆盖仅影响本次输出。

同样支持 `--auto-ottava`／`--no-auto-ottava`，统一开启／关闭所有分谱的自动八度移位；不指定时沿用项目设置，新导入默认关闭，输出命令的覆盖不写回项目。

`import` 另有 `--bpm` 覆盖速度；其他布局、动画、调号和分谱校正在桌面或 `.stproj` 中保存。`frame --time` 使用展示时间，与预览时间轴和视频帧一致：音频时间为 `max(0, 展示时间 − 开场停顿)`。“音频内乐谱零点”仍相对于原曲音频校准，不包含开场停顿。导出总时长为开场停顿加原曲音频与偏移后谱面结束的较晚者，按整帧补足；必要时补静音，保留原音频尾音。旧项目缺少新设置时使用零停顿、中速报幕和报幕常驻，已有自定义动画时长保留；旧速度显示设置被忽略，统一采用首小节滚动记号。所有命令的完整参数可用 `--help` 查看。

## 功能限制

- 目前仅支持固定速度、固定拍号。MIDI 变速／变拍、Type 2、SMPTE 时钟会明确拒绝；尚无计划支持完整的打谱编辑器或自动音频转谱功能。
- FLP 只展开所选 Arrangement 中实际放置且未静音的 Pattern；分谱身份来自 Channel Rack。未知或未验证的 Playlist 布局失败时请补充 MIDI，请勿将“解析没报错”当作实际演奏已完成对照验证。
- 不运行 FL Studio 插件。Audio Clip 的实际声音由原曲音频保留；缺少 MIDI 音符的轨道需补充烘焙 MIDI。Layer、Slide、通道琶音器、插件内部生成音符、微调与自动化有诊断或支持限制，不能保证仅凭 FLP 重建全部发声细节。
- 名称识别、自动谱号／调号及量化均为建议，其准确性需要人工判断。

有关更多信息，参见 [FLP / MIDI 导入说明](docs/importing.md)。

## 开发与依赖

Rust Vulkan 合成器保持原生渲染 ABI 兼容，传统 Qt 后端仍可显式选择。Rust 场景计算库是运行必需依赖，CPU 与 GPU 渲染统一使用它计算每帧相机、布局及活动状态。DLL 缺失或 ABI 不兼容时明确报错；GPU 回退 CPU 后仍使用 Rust 场景计算。构建、后端选择与基准方法见 [原生渲染说明](docs/rust-renderer.md)。

```powershell
uv run python scripts/build_rust.py --install
uv run pytest -q
uv run ruff check src tests scripts
uv build --wheel
```

主要流程为：导入与 `ProjectIR` → `PartMapping` → 保守识别与 music21 / Verovio 共同时序刻谱 → `CompiledScene` → Rust 场景逐帧求值 → RHI Vulkan / CPU 离屏合成 → FFmpeg NVENC / x264。预览与导出共用 `FrameRenderer`，水平图块缓存与 GPU 纹理缓存分别有容量上限。生成的 Windows x64 wheel 包含原生 DLL 与 shader，无需在安装后编译；源码首次运行或更新原生代码后须重新构建。当前尚未提供独立桌面安装程序。

## 开源协议与发布

Stavellum 采用 **GNU GPL v3 或更新版本（GPL-3.0-or-later）**，完整文本见 [LICENSE](LICENSE)。当前 FLP 导入直接使用 GPLv3 的 PyFLP，因此完整应用采用兼容的 GPL 协议。第三方库和字体保留各自许可；详见 [第三方许可声明](THIRD_PARTY_NOTICES.md)。

当前安装包元数据中的主要依赖许可证如下；FFmpeg 使用外部安装的发行版，遵循该发行版的许可证。

| 依赖 | 当前版本 | 元数据许可证 |
| --- | --- | --- |
| PySide6 | 6.11.2 | LGPL-3.0-only 或 GPL-2.0-only 或 GPL-3.0-only |
| PyFLP | 2.2.1 | GPL-3.0 |
| Mido | 1.3.3 | MIT |
| music21 | 9.9.2 | BSD-3-Clause |
| Verovio | 6.3.0 | LGPL-3.0-only |
