# 26S1_CS33-2

## 快速开始

本项目支持两种运行方式：
- 本地虚拟环境（`.venv`）
- Docker / Docker Compose

运行前请先将 `.env.example` 复制为 `.env`，并填入有效的 API Key。

```bash
cp .env.example .env
```

Windows PowerShell：
```powershell
Copy-Item .env.example .env
```

## 使用 venv 运行（推荐开发阶段）

### 1) 创建并激活虚拟环境

macOS / Linux：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

如果提示找不到 `python3.13`，可先检查本机可用版本：
```bash
python3 --version
```

Windows PowerShell：
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD：
```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

### 2) 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 3) 启动应用

```bash
python main.py --task "预测客户流失" --dataset data/bank_churn
```

`main.py` 接受自然语言任务和数据集目录或 CSV 文件。它会先运行 Reader
和 Background Reviewer，展示系统推断出的任务理解并等待确认；确认后再运行完整
workflow，默认生成 business report。常用参数：

```bash
python main.py `
  --task "预测客户流失" `
  --dataset data/bank_churn `
  --target "客户是否流失" `
  --yes
```

- `--target` 可选，可以是精确列名，也可以是自然语言目标提示。
- `--target` 也可以描述多目标任务；如果提示里已经包含精确目标列名，CLI 会保留
  完整目标范围，而不会提前压缩成单个目标列。
- `--yes` 会在理解结果通过审核后跳过确认提示。
- CLI 会在数据集目录中自动发现 `train.csv`、可选 `test.csv`、可选
  `sample_submission.csv` / `submission.csv`，以及可选说明文本。
- 对于 Kaggle 风格的多目标数据集，Reader 会优先根据
  `sample_submission.csv` 与 train/test 列差异推断目标列集合，再退回到自然语言模糊匹配。
- Reader 会把正式的目标协议写入 `state.target_spec`，其中包含解析后的目标列、
  submission 协议、评估提示以及证据链。
- 当系统解析出多个目标列时，确认界面会显示 `Target columns`，并直接沿用完整目标范围，
  不会强制用户再缩窄为单个目标列。
- 旧的 `target_column` / `target_columns` 字段仍会同步写入，但它们现在只是兼容镜像，
  主要用于尚未完成迁移的下游阶段。
- 如果没有 `test.csv`，系统会继续使用训练集内部验证，并在报告中记录该限制。

旧的 Titanic 技术报告 demo 仍可使用：

```bash
python run_multiagent.py titanic
```

## 使用 Docker 运行

Compose 会挂载本机的 `data/` 与 `artifacts/` 到容器内（路径与本地直接运行 `main.py` 一致）。请确保仓库根目录存在 `.env`（Compose 通过 `env_file` 注入变量）。

默认任务与数据集与上文示例一致；可通过环境变量覆盖：

- `MAIN_TASK` — 对应 `--task` 的自然语言描述（默认：`predict customer churn`）
- `MAIN_DATASET` — 容器内 `--dataset` 路径（默认：`data/bank_churn`）
- `MAIN_ENV_FILE` — 传给 `--env-file`（默认：`.env`）
- `MAIN_YES` — 设为 `1` 时自动加上 `--yes`（默认）；设为 `0` 则保留确认交互（需 TTY，例如 `docker compose run --rm -it app`）

### 1) 使用 Compose 构建并启动

```bash
docker compose up --build
```

### 2) 自定义一次性运行

```bash
docker compose run --rm app python main.py --task "你的任务" --dataset data/titanic_autokaggle --yes
```

### 3) 后台运行（可选）

```bash
docker compose up --build -d
```

### 4) 停止服务

```bash
docker compose down
```

## 说明

- 项目要求 Python 版本 `>=3.13`（见 `pyproject.toml`）。
- `docker-compose.yml` 会从 `.env` 注入环境变量。
- 请勿提交包含真实密钥的 `.env` 文件。
- 如果 PowerShell 阻止脚本执行，可先运行：
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
