# PR: 用 Rust 重写原生 Vulkan 合成器 + 场景求值核心，全新欢迎页

## 变更概述

用 Rust 完全重写原生渲染层，替换 Qt QRhi C++ 后端（保持 `sprhi_*` C ABI v3 逐字节兼容），
并新增场景每帧求值数学的 Rust 核心（`spcore_*` ABI v1）。桌面上以无边框深色欢迎页焕新。

## 渲染器（native/rust-renderer，ash 直连 Vulkan）

- 每 quad 一个 48 字节实例，六个角点在顶点着色器由 gl_VertexIndex 展开（替代 CPU 6×32 字节顶点，上传量 -75%）
- 相邻同纹理 quad 合并为单次实例化绘制，保持绘制顺序
- 实例/纹理上传/回读三块持久映射宿主缓冲，只增不减；回读采用紧凑行（bufferRowLength=宽度），无 CPU 除距
- 一批最多 8 帧共用一条命令缓冲、一次提交、一次 fence 等待；纹理上传随同批提交
- **HOST_CACHED 回读内存**：普通 host-visible 内存在独显上映射为 write-combined，读回只有 ~118MB/s（1080p 每帧 ~70ms）；换用缓存类型后合成基准 13.5 → 666 fps
- CPU 帧缓冲全局池复用；WGSL→SPIR-V 构建期由 naga 编译并内嵌
- 契约测试全部通过：BGRA/RGBA 字节一致、混合数学、批量一致性、错误消息逐字匹配、跨线程释放

## 场景核心（native/rust-core）

TimeAxis、相机求值、五次埃尔米特轨道采样、LayoutTimeline.at、活动包络、瓦片规划头；
Python 侧 `src/stavellum/_core.py` 序列化；黄金对拍测试 200+ 时间点全字段相对误差 <1e-9。

## 实测（RTX 5070 Laptop）

| 场景 | CPU (Qt) | Rust Vulkan | 提速 |
|---|---|---|---|
| 1080p 导出流（demo） | 234 fps | 538 fps | 2.3× |
| 4K 导出流 | 78 fps | 149 fps | 1.9× |
| 合成基准（批量8） | — | 666 fps | 44×（修复 WC 回读后） |
| 完整流水线导出（3分钟工程） | 65.8s | 38.5s | 1.71× |

## 构建与打包

- `scripts/build_rust.py [--install]` 一键构建双 crate；无 MSVC 时自动回退预装 windows-gnu 工具链（链接器由 `.cargo/config.toml` 固定，libgcc 静态链接）
- wheel 打包携带两个 DLL；`hatch_build.py`/`pyproject.toml` 同步更新
- 原 C++ 构建路径保留（`scripts/build_rhi.py`），可用 `STAVELLUM_RHI_DLL` 切换

## 欢迎页

无边框深色落地页：品牌琥珀主按钮、最近工程卡片、右侧发光谱线装饰、内页式使用指南/关于、
自定义拖拽/最小化/关闭与右下角缩放手柄；与 gui.py 的信号和状态接口完全兼容。

## 测试

- 全量 pytest：1215 通过 / 0 失败（含全部原生集成测试与欢迎页改版适配）
- cargo test + clippy（两 crate 零警告）、ruff 全过
- 新增 `tests/test_core_native.py` 黄金对拍与 `tests/test_rust_renderer.py` ABI 验证

## 说明

- 兼容性：`_rhi.py` 优先选择已安装的 `stavellum_rust.dll`；缺失时回退旧 Qt DLL 或禁用 GPU
- `rust-core` 已就绪但尚未接入 Python 热路径（`layout_at`/`commands` 仍走纯 Python），接线作为后续 PR
