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
| `encode_probe.py` | §5 交付规格的全部实测数字，含编码器内存 | 见各子命令 |
| `package_probe.py` | §7 打包体积/启动时间、冻结后进程池 | 每种配置约 90 s 构建 |

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

## 已复现的关键数字

跑 `vr-compose frame --width 3840` 应当得到 §9 的基线并 PASS：

```
median 0.68   mean 1.02   p95 3.10   std>8 = 0.25%
```

跑 `inspect_data.py` 应当得到 `46.85 s/frame`、`69.27 GiB`、`91.1 → 68.3 MiB/frame`。
跑 `coverage_map.py` 应当得到覆盖率 `2 tiles 61.19% / 3 tiles 27.53% / 4 tiles 11.25%`
与密度 `min 0.355 / p5 0.433 / median 0.781`。
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
