## 1. 铺设基础 DS 工具库 (Setup DS Toolset Library)

- [x] 1.1 创建 `app/tools/profiler.py` 并实现 `profile_dataset`，用以把任何 DataFrame 压缩提纯出 Context JSON
- [x] 1.2 创建 `app/tools/cleaning_tools.py` 并实装用于异常检测、缺失插值等数据清洗相关的纯函数工具集
- [x] 1.3 创建 `app/tools/feature_tools.py` 并组装正则表达式文本切割与值离散分层编码类型的工程函数
- [x] 1.4 为全套 DS API 上足充分严密稳定的本地 Unit 测试基准验证变动不脱靶

## 2. Pydantic 参数约定与 LLM 注入指令建立

- [x] 2.1 书写对标功能箱入参强硬绑定的 Pydantic 模型映射 (比如 `ToolCall`) 定义在 `app/schemas/tools.py`
- [x] 2.2 定义建立 `app/planners/prompts.py` 去储存能使各大模型屈服于严守 JSON 输出边界的预设 System 指令集

## 3. 安装与接驳 LLM 智能中控 Planner

- [x] 3.1 创建接盘大脑模块 `app/planners/base.py` 承载实际通过 LangChain API 对 LLM 去格式化提示词和截获还原数据等控制逻辑
- [x] 3.2 不触发扣费 API 请求下，引入 Mock 的预测策略来测压运行这套全新的 Planner 逻辑是否正常

## 4. 全面革新原有的 Graph 节点逻辑

- [x] 4.1 迁移更新 `preliminary_eda.py` ，彻底弃用原本写死的检查手段全面换装接入 `profile_dataset` 以生成客观视角画像
- [x] 4.2 清除 `data_cleaning.py` 旧有手动处理逻辑，将节点升级为直接下放配置到 Planner 以收取调用命令阵列，并且自动化依次施展 `cleaning_tools.py` 内部手段的管道系统
- [x] 4.3 `feature_engineering.py` 对齐上面的更改方法将其逻辑全权转交通过调度特征相关命令运行实现
- [x] 4.4 把所有的 Mock Planner 逻辑插满集成连带 State workflow 的综合工作验证流确保流转畅通无碍
