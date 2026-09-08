# VR-Compose

把 Unreal Engine 渲染出的 **20 路方形相机序列帧**合成为 **360° 球形全景视频**（equirectangular），
直接输出符合交付规格的 MP4。

仓库：<https://github.com/Pannic17/VR-360-Compose>

**进度**：P0–P6 已完成（rig 反解、单帧正确、序列吞吐 + 直出 MP4、收尾与母版 sink、画质、
输出命名与帧模式、GUI）。下一步 P7：交付合规与色彩管线调优。
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

### 图形界面

```powershell
vr-compose-gui
```

选来源目录、选参数、点开始。进度、日志、取消都在窗口里；
**作业跑在独立子进程**，所以界面不会卡，作业崩了也不会带走窗口。
取消之后已完成的部分保留，把日志里那条输出路径填回「输出位置」就能续跑。

### 命令行

```powershell
vr-compose --source E:/22 discover
vr-compose --source E:/22 frame --frame 1656 --out pano.png
vr-compose --source E:/22 sequence --frames 1656-2433
```

第三条就是正式渲染：输出落在程序所在目录，文件名自动生成为
`L_Cathedral_20260908_163000.mp4`（`<stem>_<日期>_<时间>`），默认 8K / H.264 / 200 Mbps / 30 fps。

**自动命名每次跑都是新名字，所以它不会续跑。** 中途 Ctrl-C 想接着跑，
把日志里那行 `output :` 的路径用 `--out` 传回去 —— 那时才按分段续跑。

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
| `--sampler` | `catmullrom` | 见下面「采样器」 |
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
| `--out` | 程序目录下自动命名 | 视频模式是 .mp4 路径，帧模式是**目录**。自动命名是 `<stem>_<日期>_<时间>`，**每次跑都是新名字，所以不会续跑**；要续跑就把上次那条路径显式传进来 |
| `--stem` | | 目录里有多个文件 stem 时选择要处理的集合 |
| `--out-format` | `mp4` | `mp4` 交付；`png` 出无损母版序列（按源的原生密度，一帧一个文件）；`exr` 预留未实现 |
| `--sampler` | `catmullrom` | 分数位置怎么读 tile：`catmullrom` 最还原（对源 +1.24 dB），`bilinear` 快 2.8 倍但更软，`nearest` 是 P1 基准/预览档 |
| `--feather-power` | `2` | 混合时对「距 tile 边缘」的指数；越高越只信最中央看到这个方向的那个 tile。实测最优 2–4 |
| `--bit-depth` | `8` | 母版位深（仅 `--out-format png`）。`16` 保住重采样的亚灰阶精度，代价是 133 MiB/帧（8 位 39 MiB） |
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
| `--progress-json` | | stdout 只输出 NDJSON 进度事件，人类可读的行改走 stderr。GUI 用的就是它，也可以拿来自己写脚本 |
| `--cancel-on-stdin` | | 收到一行输入（或 stdin 到 EOF）就干净停止，已完成的部分保留、可续跑 |

## 两种输出模式

| | 视频模式（默认） | 帧模式 `--out-format png` |
|---|---|---|
| 产物 | 一个 `<stem>_<日期>_<时间>.mp4` | 一个 `<stem>_<日期>_<时间>/` 目录，里面是 `<stem>_S_<帧号>.png` |
| 尺寸 | 按 `--size` 交付（8k/4k），需要时由 ffmpeg 降采样 | **按 input 自己的密度**（本 rig = 4× tile，即 7680×3840），**不接受 `--size`** |
| 压缩 | 走 H.264/H.265 有损压缩 | **无损**。PNG 的 zlib 等级只影响文件大小和耗时，**不改变任何一个像素**（有测试断言）；要纯存储用 `--compress-level 0`，磁盘约 2.3 倍 |
| 续跑 | 按分段（需显式 `--out`） | 按文件存在（需显式 `--out`） |

这两条限制在帧模式里是**结构性**的，不是约定：那条路径根本没有 codec / 码率 / 交付尺寸可用，
而母版宽度不等于 input 推出来的密度时会直接报错。

**母版序列细节**：一帧一个无损 PNG，文件名 `<stem>_S_<帧号>.png`，帧号沿用源的补零位数。
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

## 母版位深与磁盘

`--out-format png` 默认出 8-bit（8K 约 **39 MiB/帧**，778 帧约 30 GiB）。
加 `--bit-depth 16` 出 16-bit（约 **133 MiB/帧**，778 帧约 **101 GiB**）——
源是 8-bit，所以 16 位不带来渲染器的新信息，它保住的是**重采样与混合产生的亚灰阶精度**，
对「还要被下游再重采样一次」的母版有意义。程序启动前按位深预检磁盘空间，不够会拒绝开始。

（Pillow 写不了 16-bit RGB PNG，所以这条路径由 `io.write_png16` 自己写 PNG，用 Up 滤波。）

## 采样器（`--sampler`）

tile 到全景的映射几乎处处在**放大**（次轴缩放比 0.82 赤道 / 0.06 极冠），所以关键不是抗锯齿
而是**插值质量**。三个选项，按对源的还原能力排（把母版重采样回 tile 网格比 PSNR）：

| | 对源还原 | 8K warp | 说明 |
|---|---|---|---|
| `catmullrom`（默认） | **48.90 dB** | 10.1 s/帧 | 4×4 三次插值，最还原 |
| `bilinear` | 47.66 dB | 3.6 s/帧 | 快 2.8 倍，但更软 —— **保真度上并不比最近邻好** |
| `nearest` | 47.85 dB | 2.0 s/帧 | P1 基准，预览用；有半像素位置误差（块状） |

**注意别用「重叠一致性」（指标 A）判断画质** —— 那个指标偏爱模糊，
catmullrom 在它上面比 bilinear 差，却对源还原好 1.24 dB。细节见 AGENTS.md 第 8 节。

## 性能参考（本机：Ryzen 9 7950X，32 线程，128 GB）

| | |
|---|---|
| 8K 单帧（`catmullrom`） | ~10.1 s（默认，最还原）；16-bit 母版 ~13.1 s |
| 8K 单帧（`bilinear` / `nearest`） | ~3.6 s / ~2.0 s |
| 778 帧 8K | `catmullrom` 约 2.2 小时 / `bilinear` 约 53 分钟 / `nearest` 约 32 分钟 |
| 4K 单帧 | 与 8K 同价（按 8K 母版拼，再降采样）；`--stitch-at delivery` 更快，但会走样 |
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
.venv/Scripts/python.exe -m pytest              # 265 个测试；几何/目录发现的测试不依赖真实数据
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
  cli.py          discover / frame / sequence，以及 GUI 用的 NDJSON/取消接口
  gui/            PySide6 窗口：window.py + main_window.ui + style.qss
tests/            pytest；真实数据的检查单独标记，数据不在时自动跳过
tools/            复现 AGENTS.md 每个数字的脚本
```

## 更多文档

- [AGENTS.md](AGENTS.md) —— 相机装配、坐标约定、数据布局、交付规格与实测、性能基线、验证方法、硬约束
- [ROADMAP.md](ROADMAP.md) —— 分阶段目标、完成判据、当前进度
- [tools/README.md](tools/README.md) —— 分析与验证脚本
