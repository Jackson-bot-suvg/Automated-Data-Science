## 新增要求 (ADDED Requirements)

### Requirement: 提取通用数据画像 (Generate Data Profile)
系统必须能够生成一份高度浓缩、面向表格级数据集的统计 JSON 画像文件，画像需聚合各类结构的 schema 类型信息、缺失率以及目标分类不平衡现象。

#### Scenario: 探针工具摄取原始数据表格
- **WHEN** Initial EDA 执行节点加载目标数据集表格后
- **THEN** 生成一份描述其深层统计维度的精密 `eda_findings` JSON 信息。

### Requirement: 输出受控结构调令 (Output Structured Tool Calls)
LLM Planner 在被赋予数据集画像配置约束时，必须且只能吐出格式化的离散 `ToolCall` 调用清单，严禁输出原生包裹格式的 Python 执行脚本。

#### Scenario: Planner 基于发现生成填补策略
- **WHEN** Planner 注意到当前特征年龄存在严重缺失然而存活样本率很平均时
- **THEN** 它必定回传严谨的 JSON 结构 `{"tool": "fill_missing", "column": "Age", "parameters": {"strategy": "median"}}`
