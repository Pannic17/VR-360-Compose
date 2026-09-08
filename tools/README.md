# tools/

分析与验证脚本。**每一个都对应 [AGENTS.md](../AGENTS.md) 里的一组结论**，
目的是让那些数字可复现 —— 而不是让人只能信文档。

产品代码在 `src/vr_compose/`；这里的脚本不被 `src/` 导入，只被人和 CI 调用。
它们对 `E:\22` **只读**，任何输出都要显式给 `--out-dir` / `--work`。

## 脚本

| 脚本 | 验证 AGENTS.md 的 | 耗时 |
|---|---|---|
| `inspect_data.py` | §4 数据布局、§8 上一代性能基线、alpha、重复文件 | 数秒（`--skip-duplicates`）/ 约 1 分钟 |
| `fit_rig.py` | §3 装配反解；**也是遇到未登记布局时的求解工具** | 每相机约 30 s |
| `coverage_map.py` | §3 球面覆盖与采样密度 | 数秒 |
| `resample_probe.py` | §3「重采样到底在做什么」—— 映射的雅可比，决定 P4 的滤波器 | 数秒 |
| `linear_light_probe.py` | §8「线性光混合」—— 分别量混合与插值改到线性光的效果 | 约 1 分钟 |
| `detail_probe.py` | §8/§9「极区细节保留度」—— 源 tile 与母版按纬度的局部对比度 | 约 1 分钟 |
| `fidelity_probe.py` | §8「三个采样器」—— 母版往返回 tile 的还原 PSNR，**这才是保真度指标** | 约 2 分钟 |
| `seam_probe.py` | §8「接缝、羽化指数」—— 20 点接缝梯度不连续性 + 羽化扫描 | 约 2 分钟 |
| `temporal_probe.py` | §8「时域稳定性」—— 连续 10 帧的方差与与网格无关的跳变界 | 约 4 分钟 |
| `encode_probe.py` | §5 交付规格的全部实测数字，含编码器内存 | 见各子命令 |
| `package_probe.py` | §7 打包体积/启动时间、冻结后进程池（**探针**，用合成小程序） | 每种配置约 90 s 构建 |
| `build_exe.py` | P8 **真正的发布构建**：PyInstaller + 放入 ffmpeg + 中文说明 + 冒烟校验 | 约 2 分钟 |

已经进产品代码、不再需要独立脚本的：

- **指标 A / 指标 B** → `vr_compose.verify`，`vr-compose frame` 直接打印并给 PASS/FAIL。
  原 `stitch_probe.py` 已删除。
- **来源目录发现** → `vr_compose.source`，`vr-compose discover` 是它的入口，
  合成用例进了 `tests/test_source.py`。原 `discover_source.py` 已删除。
- **rig 定义** → `vr_compose.rig`，重复组由 `views` 派生而非硬编码。
  原 `tools/_rig.py` 已删除，这里的脚本都从包里导入。

```bash
.\.venv\Scripts\python.exe tools\inspect_data.py --skip-duplicates
```

```bash
.\.venv\Scripts\python.exe tools\coverage_map.py
```

```bash
.\.venv\Scripts\python.exe tools\resample_probe.py
.\.venv\Scripts\python.exe tools\resample_probe.py --tile 3840 --output-width 7680
```

`resample_probe.py` 是 P4 加的，回答的是 `coverage_map.py` 回答不了的那个问题：
每个输出像素上，映射是在**缩小**源（会走样，要预滤波）还是在**放大**（要好的插值核），
以及有多各向异性。**它推翻了 P4 原本的 mip/EWA 设计** —— 原生密度下两轴同时缩小的方向只占 0.1%。
注意 `coverage_map.py` 里那个 `cos³θ`「密度」是**每源像素占的立体角**，越小表示源越密，
它自己的注释和 AGENTS.md 旧文把方向读反了（P4 已更正）。

```bash
.\.venv\Scripts\python.exe -m vr_compose --source E:/22 frame --frame 1656 --width 3840
```

```bash
.\.venv\Scripts\python.exe tools\fit_rig.py --cameras 1 2 3
```

```bash
.\.venv\Scripts\python.exe tools\encode_probe.py --work %TEMP% matrix --frames 90
```

`encode_probe.py` 还有 `rate`（码率-质量曲线）、`chroma`（4:2:0 代价诊断）、
`speed`（编码吞吐）、`memory`（编码器内存 vs 线程设置）四个子命令，`--help` 有说明。

```bash
.\.venv\Scripts\python.exe tools\encode_probe.py --work %TEMP% memory
```

`memory` 是 P3 加的，对应 AGENTS.md 第 5 节「编码器内存与 64 GB 软上限」。
它用合成噪声帧而不是参考母版 —— 编码器的缓冲池按分辨率/线程/lookahead 定大小，与画面内容无关，
读 8K PNG 只会让探针更慢。**比较配置要看 commit 那一列，不要看 resident**：
同一条命令的 resident 实测在 6.5–14.8 GiB 之间跳（Windows 会修剪工作集）。

```bash
.\.venv\Scripts\python.exe tools\package_probe.py --configs excluded_onedir
```

```bash
.\.venv\Scripts\python.exe tools\build_exe.py --source E:/22
```

`build_exe.py` 与 `package_probe.py` 的分工：探针是拿一个合成的小程序去量**打包方式的代价**
（onedir 对 onefile、要不要写排除列表、冻结后进程池行不行）；
`build_exe.py` 是照那些结论**真的产出可分发的文件夹** ——
PyInstaller 走 `VR-Compose.spec`，然后把 ffmpeg/ffprobe 和中文说明放进去，
最后拿**打包好的 exe**（不是源码）跑一次两帧渲染，并与源码跑出来的**逐字节比对**。

## 已复现的关键数字

跑 `vr-compose frame --width 3840` 应当得到 §9 的基线并 PASS：

```
median 0.68   mean 1.02   p95 3.10   std>8 = 0.25%
```

跑 `inspect_data.py` 应当得到 `46.85 s/frame`、`69.27 GiB`、`91.1 → 68.3 MiB/frame`。
跑 `coverage_map.py` 应当得到覆盖率 `2 tiles 61.19% / 3 tiles 27.53% / 4 tiles 11.25%`
与密度 `min 0.355 / p5 0.433 / median 0.781`。
跑 `resample_probe.py` 应当得到原生密度下 σ_min 中位数 0.82（赤道）/ 0.06（极冠）、
两轴同时缩小的方向占 0.1%；`--tile 3840 --output-width 7680` 则是 100% 在缩小。
跑 `fidelity_probe.py` 应当得到还原 PSNR nearest 47.85 / bilinear 47.66 / catmullrom 48.90 dB
—— **注意它和指标 A 的排序相反**，指标 A 量一致性、这个量保真度，别混用。
跑 `seam_probe.py` 应当得到接缝比 median 约 0.79（<1 就是没有接缝），且各羽化指数下 p95/max 相同。
跑 `temporal_probe.py` 应当得到 nearest 四个纬度带的跳变界全部 PASS（凸核必然如此），
catmullrom 在赤道超出约 1.1%；极冠的方差比值 >1 是网格差异，不是抖动（见 §8）。
跑 `encode_probe.py matrix` 应当 12 种组合全部 `OK`，且等级为
8k H.264 6.0 / 8k H.265 6.2(6.1@100k) / 4k H.264 5.1 / 4k H.265 Main tier 5.2(5.1@28k)。
等级与档位从 `vr_compose.encode` 导入，工具里没有第二份表。
注意这个工具让 ffmpeg 自己读 PNG 并缩放，那条路径**不可复现**（AGENTS.md 第 5 节
「编码器的确定性」）—— 拿它的输出比 PSNR 可以，比文件哈希不行；管线用的是 stdin 送 raw RGB。

`--bitrates` 不给时每种尺寸用自己的档位：8k `200000,150000,100000`、
4k `57000,43000,28000`（按像素比 3.515625 缩放，bit/像素与 8k 对齐）。
等级不是查表来的 —— `select_level()` 按 H.264 Table A-1 与 HEVC Tables A.6/A.9
算出最小合规等级，所以换任意码率都能得到正确的 level/tier。

跑 `encode_probe.py memory` 应当得到 8K h264 `threads=auto` 约 **21 GiB commit**、
`sliced-threads=1:threads=16` 约 **7.9 GiB**，x265 默认约 8.3 GiB、`pools=8:frame-threads=2` 约 6.5 GiB。
整机峰值（拼接 + 编码）由 `vr-compose sequence` 自己打印，8K 实测 **23.8 GiB commit**，与总帧数无关。

跑 `package_probe.py` 应当得到 onedir `158M / 热启动 0.86s`、onefile `64M / 5.88s`，
且三种配置的 `probe` 行都以 `PROBE-OK` 开头并带 `pool=[499500, 1999000]`
（`pool=` 存在就证明冻结后的进程池没有递归重启 GUI）。

**这些数字是回归基线。** 改了几何、滤波或编码参数之后它们变了，
要么是改对了（那就更新 AGENTS.md 和这里），要么是改错了。不要默默接受漂移。

## 两个要注意的地方

- **依赖方向是单向的**：`tools/` 导入 `vr_compose`，反过来永远不行。
  这些脚本不参与产品运行，只用来复现文档里的数字。
- **`encode_probe.py` 的短片段码率不能当合规检查。** VBV 缓冲是 2 秒，
  几秒的片段平均码率合理地会超出目标；`I-gap` 为 `[]` 也只是说片段短于一个 GOP。
  码率与 GOP 的合规性要在全长渲染上验。
  P3 实测复核了这一条：60 帧 @30fps 只有 1 个 I 帧，`i_intervals` 是空集合，
  于是 GOP 检查**空过**（`conformance: OK` 但什么都没验到）。真要验 GOP 至少要 3 个 I 帧
  —— 30 fps 用 130 帧，60 fps 用 250 帧，配 `--segment-gops 1`。
