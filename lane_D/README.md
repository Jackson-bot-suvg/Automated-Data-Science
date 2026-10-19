# lane_D_新：提交目录

**状态：无需 API 的准备工作已完成；真实 DeepSeek 实验尚未启动。**
只需提交这个文件夹。它直接使用项目已有的 `app/` 和 `data/`，不包含第二份仓库、数据、虚拟环境或压缩包。

## 基于哪个 main

已核对用户提供的 `Automated-Data-Science-main`：117 个 app/data 文件与原实验基线完全一致。此 main 是源码压缩包，没有 `.git`；运行器改用逐文件 SHA256 校验，不再要求 Git 元数据。

支持两种放置方式：

- 当前方式：`lane_D_新/` 与 `Automated-Data-Science-main/` 同级，自动找到 main。
- 提交到项目：把整个 `lane_D_新/` 放进 main 根目录，自动使用父目录的 app/data。

其他位置请设置 `LANE_D_BASE=/absolute/path/to/Automated-Data-Science-main`。

## 目录里保留什么

| 内容 | 用途 |
|---|---|
| `paper/evidence_report.pdf` | 唯一一份 5 页中文准备报告，沿用此前已验证版本；其中旧包路径以本 README 为准 |
| `scripts/`、`src/`、`tests/` | 准备、运行、续跑、观测、判分、统计及测试源码 |
| `protocol/`、`cases/`、`FREEZE_SIGNOFF.yaml` | 数据/代码哈希、方法定义、预注册计划、描述变体及未签字快照 |
| `logs/plan_ledger.csv` | 144 格 DEV：8 数据集 × 3 描述 × 2 过程随机块 × 3 方法 |
| `logs/formal_plan_ledger.csv` | 72 格正式 D0 和 432 格 D1，全部未启动 |
| `logs/smoke/`、`results/` | 必要离线执行证据、输出、测试和整理验证记录 |
| `env/` | 依赖锁定、运行环境信息和空白 API 配置示例 |

不保留：重复仓库、原始数据、`.git`、虚拟环境、临时训练文件、模型缓存、预览图片、重复 ZIP/PDF、整本任务书副本、重复全文状态、没有实跑数据的空结果表。空结果表可由 `--rebuild` 重建。

## 如何运行

Python 3.13；在新环境中安装 `env/environment.lock`。macOS 需要已有 OpenMP/libomp。环境放在提交目录之外：

```bash
python3.13 -m venv ../.venv_lane_d
../.venv_lane_d/bin/python -m pip install -r env/environment.lock
export LANE_D_PYTHON="$PWD/../.venv_lane_d/bin/python"
bash reproduce.sh --dry-run
bash reproduce.sh --smoke
bash reproduce.sh --rebuild
```

`--smoke` 不访问 API，在原始 Titanic、Ghouls 上检查观测开关输出一致性，并运行测试。临时观测帧在 `tmp/`，不需提交；此前的原始帧未重分发，可从 main 的原始数据重新生成。`--rebuild` 生成的空结果/统计表不表示发生了真实模型实验。

API 配置补齐之后：

```bash
export LANE_D_ENV_FILE=/absolute/path/to/private.env
"$LANE_D_PYTHON" scripts/run.py --pilot --limit 2
bash reproduce.sh --rebuild
# 核验实际响应模型 ID、动作证据和预算后继续
bash reproduce.sh --resume
```

不要把密钥文件提交。计划模型别名为 `deepseek/deepseek-flash`，用户要求的 DeepSeek 4.1 Flash 身份仍需服务响应核验。`--resume` 不覆盖或自动重跑已启动格。

## 结论边界

真实模型运行数仍为 **0**，错误率为 **NA**，不是“0/144”。离线工作流测试不计入自然错误分母。F0 与 M2 的检查范围重合，不能据随机差异声称门控增量；M0 是 reviewer 层消融。过程种子不等于改变了原稿固定的训练种子。原始测试标签不可得，不报告独立测试准确率。正式确认和迁移仍待签字、任务冻结与第二宿主/模型条件。

这次只整理提交内容并适配所给 main；没有改动 main、没有改变研究结论、没有执行真实 API 调用。共享协议标识为 RTS-TAI-PARALLEL-v4.0；完整任务书是用户提供的 4.pdf，未重复打包。
