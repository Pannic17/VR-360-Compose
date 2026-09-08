# VR-Compose

把 Unreal Engine 渲染出的 **20 路方形相机序列帧**合成为 **360° 球形全景视频**（equirectangular），
直接输出符合交付规格的 MP4。

仓库：<https://github.com/Pannic17/VR-360-Compose>

**进度**：P0（rig 反解）、P1（单帧正确）、P2（序列吞吐 + 直出 MP4）、P3（收尾与母版 sink）已完成，
8K 实测 **2.0 s/帧**（上一代 46.85 s）。下一步 P4：画质（各向异性滤波、极区重建、线性光混合）。
阶段目标与所有实测数字见 [ROADMAP.md](ROADMAP.md)。

---

## 它做什么

| 输入 | 输出 |
|---|---|
| 一个来源目录：`Camera1..Camera20`，每个目录里是 1920×1920 的 PNG 序列帧 | 一个 MP4：7680×3840（或 4096×2048）等距圆柱全景，H.264 或 H.265，30/60 fps |

相机装配是固定的（5 个方位扇区 × 3 个仰角、90° 方形视场、共节点），所以不需要标定、
不需要特征匹配 —— 合成就是一次固定重采样加羽化混合。20 个文件里只有 15 个是不同视角，
程序只读那 15 个。

## 环境要求

- Windows，Python **3.13**
- **ffmpeg + ffprobe**，带 `libx264` 和 `libx265`（放在程序同级目录，或在 PATH 上；
  程序先找同级目录）
- 内存：8K 一次作业实测峰值 **23.8 GiB**（拼接侧 3.7 + ffmpeg 20.1，已提交字节），
  **与总帧数无关**（60 帧和 180 帧完全一样）。**不设内存预算，软上限 64 GB，超了只打 Warning，
  不会中断。** 内存紧的机器加 `--encoder-threads 16`：实测峰值降到 **9.8 GiB（−59%）**，
  吞吐只掉 4%（2.24 → 2.33 s/帧）
- 磁盘：8K 200 Mbps 的 778 帧约 2 GB；程序启动前会检查空间，不够会拒绝开始

## 安装

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

装好后 `.venv/Scripts/vr-compose.exe` 就是命令入口（下文简写为 `vr-compose`）。
打包成独立 exe 是 ROADMAP 的 P8。

## 快速开始

```powershell
vr-compose --source E:/22 discover
vr-compose --source E:/22 frame --frame 1656 --out pano.png
vr-compose --source E:/22 sequence --frames 1656-2433
```

第三条就是正式渲染：输出落在程序所在目录，文件名自动生成为
`L_Cathedral.1656-2433.7680x3840.h264.mp4`，默认 8K / H.264 / 200 Mbps / 30 fps。
中途 Ctrl-C 了，**原样再跑一遍**就从断点续上。

## 命令详解

### `discover` —— 看看程序找到了什么

```powershell
vr-compose --source E:/22 discover
```

报告检测到的相机数、tile 尺寸、共有的帧范围、上一代输出目录，以及匹配到的 rig。
来源目录不合规时会逐条列出原因（编号不连续、tile 非方形、各相机帧数不齐……），而不是一句"无效"。

`--source` 可以省略：程序会在自己所在目录、它的子目录、上一级目录的子目录里按目录结构自动找。

### `frame` —— 拼一帧并自检几何

```powershell
vr-compose --source E:/22 frame --frame 1656 --width 7680 --out pano.png
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--frame` | 第一个共有帧 | 帧号 |
| `--width` | 源的原生密度（本 rig = 4× tile，即 1920 → 7680） | 输出宽度，高度自动为一半 |
| `--out` | 不写 | 写 PNG 到这里 |
| `--stem` | | 目录里有多个文件 stem 时选择要处理的集合 |

除了图，它还打印**几何自检**（重叠区一致性，AGENTS.md 里叫指标 A）：15 个 tile 在重叠处互相之间
的亮度差。正确的 rig 下 median 约 0.7/255；超过 1.0 就 FAIL，退出码非 0。这比肉眼看缩略图可靠 ——
球面是对称的，上下颠倒的拼接在缩略图上看起来完全正常。

### `sequence` —— 直出交付 MP4

```powershell
vr-compose --source E:/22 sequence --frames 1656-2433 --size 8k --codec h264 --bitrate high
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--frames` | `all` | `1656-2433`、`1656-`、`-1700`、`1,5,9`；范围会裁到实际存在的帧 |
| `--size` | `8k` | **交付**尺寸：`8k` = 7680×3840，`4k` = 4096×2048 |
| `--codec` | `h264` | `h264` / `h265` |
| `--bitrate` | `high` | 8k 档：high 200 / mid 150 / low 100 Mbps；4k 档：57 / 43 / 28 Mbps |
| `--fps` | `30` | `30` / `60`，GOP 随之为 60 / 120 帧（2 秒） |
| `--out` | 程序目录下自动命名 | 输出 .mp4 路径 |
| `--stem` | | 目录里有多个文件 stem 时选择要处理的集合 |
| `--out-format` | `mp4` | `mp4` 交付；`png` 出无损母版序列（按源的原生密度，一帧一个文件）；`exr` 预留未实现 |
| `--stitch-at` | `native` | `native` 按源的原生密度拼母版（本 rig = 4× tile），再由编码器 Lanczos 降到交付尺寸；`delivery` 直接拼到交付尺寸 —— 快，但会走样，**只用于预览**（见下） |
| `--deterministic` | 关 | 续跑结果逐字节一致；编码器慢 6–8 倍，一般不用（见下） |
| `--segment-gops` | `5` | 每个可续跑分段含几个 GOP（5 × 2 s = 10 s） |
| `--stats-every` | `100` | 每多少帧做一次几何自检 |
| `--decode-workers` | `4` | 解码线程；**多于 4 会拖慢 warp**（GIL 争用，实测） |
| `--warp-threads` | `8` | 重投影线程 |
| `--encoder-threads` | 不设 | 给编码器的线程上限，用来压它的内存：不设时整机峰值约 24 GiB、最快；给 16 降到约 10 GiB，吞吐掉 4%。只是脚印旋钮，**不影响可复现性** |
| `--no-resume` | | 忽略已完成的分段，全部重编 |
| `--keep-segments` | | 合并后保留分段文件 |
| `--no-bar` | | 不显示进度条，只输出普通日志行 |

**母版序列**：`--out-format png` 时 `--out` 是**目录**，一帧一个无损 PNG，按源的原生密度
（本 rig = 4× tile），文件名沿用源的编号（`L_Cathedral.1656.png`）。
实测 8K **2.40 s/帧、39.6 MiB/帧**，778 帧约 30 GiB —— 是同长度 MP4 的 15 倍，所以它是按需产出的
（P4 比画质、归档），不是交付路径。按文件存在续跑，`--compress-level` 只影响文件大小不影响像素。
这时候 `--codec / --bitrate / --fps / --size` 之类会**明确报错**而不是被忽略。

**输出规格**（用户确认的交付规格，全部由 `ffprobe` 逐项校验）：MP4 容器、`avc1` / `hvc1`、
High / Main profile、`yuv420p` 8-bit 4:2:0、**只有 I 帧和 P 帧（B=0）**、闭合 GOP 2 秒、无音轨。
level 由规范表算出最小合规值，不交给编码器（编码器自选会给出 HEVC 里不存在的 Level 7.1）。

**续跑**：序列按整 GOP 切成分段，每段一个独立的 ffmpeg 进程，最后无损拼接。中断后重跑同一命令，
已完成的分段直接跳过（用 `ffprobe` 校验帧数，不是看文件在不在），只补缺的。

**几何自检**：每 `--stats-every` 帧检查一次重叠一致性，FAIL 就立刻停 —— 装配错了不会浪费几个小时。

**进度**：tqdm 进度条，显示已完成帧数、当前分段、每帧秒数、相对上一代 46.85 s/帧 的倍数。
每段结束打印各阶段耗时分解（等解码 / warp / 几何门 / 写管道）。

## 来源目录约定

```
<来源目录>/
  Camera1/[任意一层子目录/]<stem>.<帧号>.png
  Camera2/...
  ...
  Camera20/...
  FinishTaskOutput/        （可选，上一代的输出）
```

- 相机目录名 `Camera<数字>`，大小写不敏感，允许 `Camera_1` / `Camera 1`
- 子目录名随意（参考数据里叫 `Tempory`），也可以没有
- 文件名 `<stem>.<帧号>.png`，帧号零填充位数一致；不同 stem 视为不同的集合（例如不同场景、版本或命名前缀）
- 只有**所有相机都有**的帧才会被处理
- **相机数必须是已登记的装配**（目前只有 20 路）。别的数量会明确拒绝，并提示用
  `tools/fit_rig.py` 对着一张参考全景反解

## 母版尺寸与交付尺寸

**母版跟着源走，交付跟着规格走。** 母版宽度 = `tile_size × 360 / fov`（本 rig = 4× tile），
即 1920 的 tile 对应 7680×3840 —— 这个宽度正好等于 tile 中心的像素密度（21.33 px/度）。

- `--size 8k` 配 1920 的 tile：母版就是交付尺寸，**什么都不重采样**。
- `--size 4k`：拼 7680×3840 母版，由 ffmpeg 在一个 scale 滤镜里 Lanczos 降到 4096×2048。
  母版不落盘。实测这样比直接拼 4096×2048 **对母版的还原好 2.52 dB**，
  而直接拼的版本高频反而多 31%（那是走样，不是细节）、帧间差多 5.1%（头显里的爬行）。
- 将来源头按 **16K 渲染**（3840 的 tile）而交付 8K：同一条路径，母版 15360×7680、交付 7680×3840，
  不用改任何参数 —— `--size` 保持 `8k` 就行。注意 16K 只能当母版，
  H.264/HEVC 都没有等级容得下 15360×7680。
- `--stitch-at delivery` 是直接拼交付尺寸的快速预览档，**不要用它出片**。

## 性能参考（本机：Ryzen 9 7950X，32 线程，128 GB）

| | |
|---|---|
| 8K 单帧 | ~2.0 s（warp 1.84 s，解码完全藏在后面） |
| 778 帧 8K | ~26 分钟 |
| 4K 单帧 | 与 8K 同价（按 8K 母版拼，再降采样）；`--stitch-at delivery` 约 0.6 s，但会走样 |
| 上一代流程 | 46.85 s/帧，同样 778 帧要 10 小时 |
| 8K 编码 | 不是瓶颈：x264 只用到 0.4 核 |

GPU 加速列为可选项（估计 warp 可到 30–40 ms，778 帧约 3 分钟），未决定是否引入，见 ROADMAP。

## 常见问题

**`--frames` 别省成 `all`。** 参考数据里有一张孤零零的 `0000` 帧，`all` 会把它算进去，视频开头会跳帧。
显式写范围（`--frames 1656-2433`）。

**`ffmpeg and ffprobe were not found`** —— 把 `ffmpeg.exe` 和 `ffprobe.exe` 放到程序同级目录，或加进 PATH。
带 `libx264` / `libx265` 的构建才行。

**`no rig registered for N cameras`** —— 相机数不是 20，程序不知道这套装配的朝向。
用 `tools/fit_rig.py` 对一张参考全景反解，把结果加进 `vr_compose.rig.REGISTRY`。

**`tiles are not square`** —— 已登记的 rig 假设方形 90° 视场，长方形 tile 是另一套装配。

**`N source sets ... Choose one with --stem`** —— 目录里同时有多个文件 stem，程序不会替你猜要处理哪一组。

**`Refusing to start`（磁盘）** —— 输出目录所在盘的空间不够估算的产物大小（分段 + 合并文件 + 余量）。

**Ctrl-C 之后怎么办？** 原样再跑一遍就续上。取消在**帧边界**生效，所以按下之后最多再等一帧
（8K 约 2 s）才退出，退出码是 130；这样不会有半帧写进编码器。急的话再按一次 Ctrl-C 立刻中断。
完成的分段一定保留，`.part` 一定清掉。

**续跑出来的文件和一次跑完的不一样？** —— 默认模式下两者**结构相同**（帧数、GOP、合规性全一致），
x264 在长分段上通常也逐字节一致，但末尾的短分段不保证；x265 在多线程下本来就不可复现。
要逐字节一致就加 `--deterministic`（编码器慢 6–8 倍）。原因与实测见 AGENTS.md 第 5 节。

## 开发

```powershell
.venv/Scripts/python.exe -m pytest              # 228 个测试；几何/目录发现的测试不依赖真实数据
.venv/Scripts/python.exe -m ruff check .        # lint
.venv/Scripts/python.exe -m ruff format .       # 格式
.venv/Scripts/python.exe -m mypy                # strict，覆盖 src / tests / tools
```

四样全绿才算过。改了几何、滤波或编码参数后，跑 `tools/` 里的脚本复现 AGENTS.md 的数字 ——
它们是回归基线。

## 目录结构

```
src/vr_compose/
  projection.py   等距圆柱 <-> 针孔几何，纯数学
  rig.py          装配定义与注册表；未登记布局明确拒绝
  source.py       按目录结构发现来源集
  stitch.py       逐帧重投影 + 羽化混合（基准实现）
  warp.py         静态 LUT，与 stitch 逐字节一致，任意线程数
  encode.py       ffmpeg 定位、level/tier 计算、分段写入、探针、拼接
  pipeline.py     解码预取 -> warp -> 分段编码 -> 拼接，可续跑
  verify.py       几何自检（指标 A）与环绕接缝检查
  memory.py       峰值内存测量与 64 GB 软上限告警
  io.py           读 tile（丢掉无用的 alpha）、写 PNG
  cli.py          discover / frame / sequence
tests/            pytest；真实数据的检查单独标记，数据不在时自动跳过
tools/            复现 AGENTS.md 每个数字的脚本
```

## 更多文档

- [AGENTS.md](AGENTS.md) —— 相机装配、坐标约定、数据布局、交付规格与实测、性能基线、验证方法、硬约束
- [ROADMAP.md](ROADMAP.md) —— 分阶段目标、完成判据、当前进度
- [tools/README.md](tools/README.md) —— 分析与验证脚本
