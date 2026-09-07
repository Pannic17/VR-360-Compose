# VR-Compose

把 Unreal Engine 渲染的 20 路方形相机序列帧合成为 360° 球形全景视频（equirectangular）。需要 Python 3.13+ 与 ffmpeg。

仓库：<https://github.com/Pannic17/VR-360-Compose>

**进度**：P0（rig 反解）、P1（单帧正确 + 来源目录抽象）已完成；下一步 P2（序列吞吐 + 直出 MP4）。
阶段目标与已验证的数字见 [ROADMAP.md](ROADMAP.md)。

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Usage

```powershell
.venv/Scripts/python.exe -m vr_compose discover
.venv/Scripts/python.exe -m vr_compose --source E:/22 frame --frame 1656 --width 7680 --out pano.png
```

`--source` 可选：不给的话在 exe 同级按目录结构自动查找来源集。
`discover` 报告检测到的相机数、tile 尺寸、帧范围和匹配的 rig。
`frame` 拼一帧并自己判定几何是否正确（指标 A，PASS/FAIL 决定退出码）。

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest        # tests
.\.venv\Scripts\python.exe -m ruff check .  # lint
.\.venv\Scripts\python.exe -m ruff format . # format
.\.venv\Scripts\python.exe -m mypy          # type check
```

## Layout

```
src/vr_compose/   package source (src layout)
tests/            pytest suite
pyproject.toml    project metadata + tool config
```

## 项目上下文

- [AGENTS.md](AGENTS.md) — 相机装配、坐标约定、数据布局、性能基线、验证方法
- [ROADMAP.md](ROADMAP.md) — 分阶段目标与完成判据

- [tools/README.md](tools/README.md) — 复现上述结论的分析与验证脚本
