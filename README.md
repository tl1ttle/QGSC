# QGSC

Quantum compilation with a parameterized universal gate set for XX+YY-coupled hardware.

## 文件结构

```text
QGSC/
  run.py                   命令行入口
  qgsc/code/               核心流程：预处理、搜索/路由、参数求解、保真度
  qgsc/core/               门数估计与 PDBLS 实际分解
  qgsc/monodromy/          当前使用的几何计算依赖
  benchmarks/original/     原始 QASM 输入（58 条目录快照）
  benchmarks/reqisc_processed/  对应 ReQISC 预处理版本
  examples/                小规模运行示例
  configs/smoke.json       快速验证配置
  docs/                    整理记录及文件溯源
  outputs/                 新运行结果（运行时创建，Git 忽略）
```

## 安装

当前工作环境为 Python 3.10.11；依赖版本记录在 requirements 文件中。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# 如果运行保真度模拟，还需要：
python -m pip install -r requirements-fidelity.txt
```

另外安装 LRS，并指定实际可执行文件路径（或将 lrs 加入 PATH）：

```powershell
$env:LRS_PATH = '安装目录\lrs.exe'
```

默认完整流程启用 ReQISC 预处理，需要另行准备 ReQISC/Regulus 源码、按该项目说明安装依赖并准备模板文件 `artifact/scripts/regulus_template_library.pkl`：

```powershell
$env:REQISC_ROOT = 'ReQISC源码目录'
```

如果当前环境不允许写入用户配置目录，可在运行前将配置目录放到本项目中：

```powershell
$env:XDG_CONFIG_HOME = Join-Path (Get-Location) 'outputs/runtime-config'
```

核心算法参数保留在 `qgsc/code/main.py` 的 `PARAMETERS` 中。

## 快速验证

```powershell
python run.py --help
python run.py --show-config
python run.py --config configs/smoke.json --input examples --output outputs/smoke
```

快速验证保留 ReQISC 预处理，关闭保真度模拟；需要先配置 ReQISC 和 LRS。COBYLA 迭代数仅设为 2，不用于报告正式实验性能。

## 完整流程

```powershell
python run.py --topology chain --input benchmarks/original --output outputs/chain --reqisc-root '你的ReQISC源码目录'
python run.py --topology grid --input benchmarks/original --output outputs/grid --reqisc-root '你的ReQISC源码目录'
python run.py --topology alltoall --input benchmarks/original --output outputs/alltoall --reqisc-root '你的ReQISC源码目录'
```

`--no-fidelity` 可跳过实际分解和保真度模拟；`--max-files 1` 可先运行一条线路；`--set NAME=VALUE` 覆盖其他参数。`python -m qgsc` 也可作为入口。

完整的模块分工：

| 文件 | 功能 |
|---|---|
| preprocessing.py | QASM 适配、ReQISC 前端与 2Q 操作提取 |
| decomposition_routing.py | 候选门搜索、拟分解与拓扑路由 |
| parameter_solver.py | 固定门数的参数求解、构建物理线路 |
| fidelity.py | 理想/含噪模拟与 Hellinger 保真度 |
| pipeline.py | 批处理、阶段调度和结果写入 |
| common.py / models.py / io_utils.py / runtime.py | 公共函数、数据类型、输入输出与配置 |

数据范围与整理修改见 docs/PACKAGING.md。第三方来源见 THIRD_PARTY_NOTICES.md；