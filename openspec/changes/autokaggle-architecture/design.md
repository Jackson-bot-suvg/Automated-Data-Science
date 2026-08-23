## 业务上下文 (Context)

Titanic 流水线当前是硬编码节点模式（例如：强制写入代码丢弃 `Cabin` 列、强制从 `Name` 提取 `Title`），虽然对特定数据集有效，但在遇到任意类型的陌生数据集时束手无策。原始的 AutoKaggle 框架利用 LLM 自动编写 Pandas 代码并通过 subprocess 执行来解决这个泛化问题，但这带来了严峻的安全隐患、代码崩溃问题以及单元测试的“不可能”。本设计方案提出了一套基于「原生 Tool Calling」技术的纯净解法。

## 目标与非目标 (Goals / Non-Goals)

**目标 (Goals):**
- 将系统框架从“硬规则/代码生成”的范式平滑过渡为高度确定性的“多 Agent + 工具调用”架构。
- 引入确定性的底层 Python 数据科学工具集 (MVP 构建 ~10-15 个核心方法)，涵盖特征侧写、清洗、插值变换、文本正则。
- 完整保留当前的 LangGraph `GraphState` 状态流，只修改节点 (Node) 内部运转逻辑。
- 让新增的底层工具函数与 LLM Planner 能够被无障碍地进行独立单元测试（解除代码和 LLM 调用的强耦合）。

**非目标 (Non-Goals):**
- 本期不支持文本生成序列、非结构化 NLP，或计算机视觉流水线的操作支持。
- 不替换 LangChain/LangGraph 基座逻辑。

## 关键设计推演 (Decisions)

**决策 1: 纯原生 Python 底层函数工具箱方案**
- **Rationale:** 废除 AutoKaggle 用黑盒 LLM 生成 subprocess 代码的危险做法。我们构建纯粹确定性、边界安全的底层 Python API（如 `fill_missing(df, col, method)`），这些都是不需要启动 LLM 就能单独验证并在极短时间内完成调用的死逻辑纯函数。

**决策 2: LLM 被弱化为在严格工具箱下“只做调度”的 Planner**
- **Rationale:** 利用 `pydantic` 生成规范化的工具入参模型。LLM 只有访问这些受限接口并提供结构化参数 (JSON Tool Calls) 的权限。杜绝了 LLM 因为乱抛出语法错误引起执行阻断的问题。

**决策 3: "数据特征提取器 (Data Profiler)" 作为向 LLM 提供信息的信使**
- **Rationale:** LLM 直接通过 `df.head()` 会出现视网膜盲区。为确保智能规划逻辑能够成立，我们编写特供的 `profile_dataset` 探针程序，先从庞杂 DataFrame 提炼高密度的总结信息 JSON（计算包含缺失率、类别基数以及分布偏度等静态参数），LLM 只有基于这些科学依据，才能给出精确不作祟的处理措施。

## 风险与妥协 (Risks / Trade-offs)

- **风险:** LLM 可能会产生幻觉导致引用了不存在的列名称，或提供了不支持的参数名。
- **缓解机制:** Pydantic 强大的边界验证能直接把非预期格式从底层过滤出去。当节点发现验证报错或者 Schema 不吻合时，直接将明确异常信息重走 LLM 或者自动跳过，绝对不会污染数据集本身。
