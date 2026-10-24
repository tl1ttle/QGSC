# 论文绘图代码与实验数据

本目录以当前 REVTeX 稿件为准，只整理其中实际使用的 12 张图、4 张表及其数据，不包含已经删除的图、旧版实验或未使用的线路结果。文件名沿用论文工程名称；**实际图号以此处对照表为准**。

## 目录

- `plotting/`：当前图片的 Python 绘图/排版脚本、TikZ 独立编译入口及必要公共依赖。
- `figures/`：论文实际引用的 LaTeX 插图代码及可编辑 Drawio/TikZ 源文件。
- `assets/`：当前论文使用的原始矢量 PDF，以及重新生成的图像。
- `inputs/`：图 7、图 8 排版脚本所需的矢量几何输入，不是额外实验数据。
- `data/`：论文使用的实验结果，按线路筛选并导出为 CSV；没有把整批历史输出直接复制过来。
- `tables/`：表 III、IV 的最终 LaTeX 源文件。表 I、II 与图 2、4 的插图代码同文件保存。
- `metadata/`：来源清单、图号映射和数据校验记录。

## 图号对照与运行方式

在 QGSC 根目录执行。Python 绘图依赖使用 `python -m pip install -r requirements-plotting.txt` 安装。

| 论文图号 | 内容 | 源代码/源文件 | 数值数据 |
|---|---|---|---|
| 1 | 指令集操作原语 | `plotting/generate_fig01_from_drawio.py` | 脚本中定义的示例线路 |
| 2 | Weyl 腔室 | `figures/fig02_weyl_revised.drawio` | 门坐标见同名 LaTeX 文件中的表 I |
| 3 | CX 门的覆盖区域 | `plotting/plot_fig03_coverage.py` | 脚本中的示意几何坐标 |
| 4 | AC2 腔室 | `figures/fig04_ac2_revised.drawio` | 谱坐标见同名 LaTeX 文件中的表 II |
| 5 | 分解模板 | `plotting/plot_fig05_decomposition.py` | 示意图，无实验采样数据 |
| 6 | QGSC 编译框架 | `figures/fig06_qgsc_workflow_diagram.tex` | 示意图 |
| 7 | 逻辑线路与加权 DAG | `figures/fig08_circuit_and_dag_revised.drawio`、`plotting/reflow_fig07_cxy.py` | 示意线路；矢量输入在 inputs |
| 8 | 有效路由进展 | `figures/fig09_routing_progress_revised.drawio`、`plotting/fig08_bodyfont.py` | 示意线路；矢量输入在 inputs |
| 9 | QFT3 编译示例 | `figures/fig10_panel_a_v2.drawio`、`fig10_panel_b_v2.drawio`、`fig10_panel_c_v2.drawio` | Drawio 中的示例门与参数 |
| 10 | 超幅度比例与时间降低率 | `plotting/plot_fig10_clear.py` | `data/alltoall_60.csv`、`alltoall_summary.json` |
| 11 | 全连通保真度 | `figures/fig13_fully_connected_fidelity.tex`，原生 PGFPlots | `data/fully_connected_fidelity_28.csv`；绘图坐标也保留在 TeX 中 |
| 12 | 受限拓扑性能 | `plotting/plot_restricted_dual_axis.py` | `data/restricted_topology_30.csv` |

```powershell
python paper/plotting/reproduce.py --figure 1 3 5 7 8 10 12
python paper/plotting/reproduce.py --figure 6 11 --xelatex '你的TeX安装目录/xelatex.exe'
python paper/plotting/reproduce.py --figure 2 4 9 --drawio '你的Drawio安装目录/draw.io.exe'
```

也可使用 `--all`，同时指定 `--drawio` 和 `--xelatex`。Drawio 图应通过其导出功能生成，不能把原生绘图源误当成 Python 脚本。导出的 PDF 为完整源图；论文中的裁剪、缩放、子图组合由 `figures/*.tex` 保留。图 6、11 可通过对应的 `plotting/figure_*_standalone.tex` 单独编译。

Times New Roman/Arial 与论文排版一致。在其他系统上请安装相同字体；Python 脚本在未找到 Times 时使用替代字体，图形排版可能略有差异。图 11 仍以原始 PGFPlots/TeX 源为绘图依据，CSV 是从该最终源文件逐点导出的数据副本。

## 数据与表格

| 文件 | 对应图表 | 范围 |
|---|---|---|
| `alltoall_60.csv` | 表 III、IV，图 10 | 60 条线路；传统门集与 UGS 时间/门数、ReQISC 时间、控制修正前超幅度比例 |
| `table03_family_evolution_times.csv` | 表 III | 31 个系列及 Overall 行，与论文显示的四舍五入值一致 |
| `table04_instruction_set_summary.csv` | 表 IV | 6 种门集的平均分解数、平均时间与相对 CX 的降低率 |
| `fully_connected_fidelity_28.csv` | 图 11 | 28 条线路 × 6 种方法，共 168 个保真度值 |
| `restricted_topology_30.csv` | 图 12及其正文 | 30 条线路、Chain/Grid；时间只保留 SQiSW/B/UGS，保真度只保留 SQiSW/UGS |
| `restricted_topology_family_means.csv` | 图 12 | 由上述 30 条线路按系列分组计算的绘图数据，运行图 12 脚本时生成 |

`alltoall_60.csv` 的时间单位为 g^(-1)，`count_*` 是实际分解后的 2Q 门数，`target_2q` 是公共前端的目标 2Q 门数，不是原始输入线路门数。表 III 中的原始输入门数保留在 `table03_family_evolution_times.csv`。

`restricted_topology_30.csv` 的 `chain_time_*` / `grid_time_*` 沿用原图字段名，实际值为**相对于 CX 的演化时间降低率（%）**，不是绝对时间；`*_HF_*` 是 Hellinger 保真度。负时间降低率保留，图中仅在分组取平均后将负柱高显示为零，均值仍使用带符号数据。

表 IV 的平均分解数是逐线路 `count/target_2q` 后取平均。时间降低率按平均时间之比计算；图 10 每个点的降低率则按该线路的配对时间计算。图 12 的均值线按原始 30 条线路取平均，不对 18 个系列无权重平均。

这些是当前论文分析和绘图实际使用的整理结果，不包含未被采用的保真度结果、其他基准线路的输出或完整历史日志。

## 本次验证

图1、3、5、7、8、10、12的Python脚本已运行，输出图像与论文原矢量图逐像素一致；图6、11的独立TeX入口已编译通过。Drawio命令行导出在当前环境未完成，因此图2、4、9保留论文实际使用的PDF及可编辑源文件，未声称完成重新导出验证。完整结果见 `metadata/verification.json`。
