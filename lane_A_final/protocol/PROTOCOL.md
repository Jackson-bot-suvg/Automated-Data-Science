# RTS-TAI-PARALLEL-v4.0 - protocol text (lane A copy)

Source: lane-A task PDF '1.pdf', SHA256 48f3b123c2731ede027433d43419d4e024b05e3af260c1db143715b511658def (11 pages, dated 2026-10-10, PRE-FREEZE).
Text below is a machine extraction (pypdf) of all 11 pages; Part 1 and Part 3 are the common protocol, Part 2 is lane A's own section.

```text

<!-- page 1 -->
RTS-MA-DS / TAI：负责人 A 独立实验任务书（可直接派发）
专项任务：A：审计原稿门控是否真实约束训练/预测，完成同前缀局部因果干预。
共同协议：RTS-TAI-PARALLEL-v4.0；文档日期：2026-10-10；状态：PRE-FREEZE。 此 PDF 完整包含四人
共同协议、本人专项实验与共同交付要求，可离线独立阅读，不需要同时阅读其他三个 PDF。
请先确认： 原稿研究与既有 8 数据集来自用户提供的《Contract-Gated Multi-Agent Workflows for Localizing
Target/Submission Drift in Automated Data Science》（稿件第 3–7 页）；12 CONF 和 4 TRANSFER 的具体
dataset ID、版本、SHA 和机器预算是尚待团队签字的固定字段，不可在签字前声称正式确认。实验“质量合
格”“结果可用”“有利创新”是三件不同的事。
第一部分　四人共同且完全一致的公开／固定实验协议
1. 协议身份、适用范围与签字状态
本节是四份 PDF 的共同协议正文，内容和版本必须完全一致。项目代号 RTS-TAI-PARALLEL-v4.0，制定日期
2026-10-10，状态：待签字冻结（PRE-FREEZE）。它是一份可直接执行的研究任务规格，不表示 12 个新增数
据集已经选定、代码已完成或实验已经通过。团队负责人须在正式 CONF 运行之前补全本协议第 11 节签字清单
及任务登记信息，形成带 SHA256 的 FREEZE_SIGNOFF.yaml；每个人的结果包必须附同一份签字快照。
原稿研究的任务包为 P=(q,D_train,D_test,S,H)，契约为 C=(Y,f,m,I,O,Σ_S)：训练目标 Y、任务类型 f、指标 m、
行标识 I、输出角色 O 和提交模式 Σ_S。它声称通过 target_spec、语义对齐与 approved/retry/fatal reviewer
gate 减少语义漂移，重试预算每阶段最多 3 次。论文中的 reviewer gate 与这里的科研实验质量门控是两种不
同概念。
原稿已经报告：8 个干净任务，各重复 3 次，补丁后 24/24 有效；16 个受损任务包中有 7 个提前拦截；消融 No
reviewer gates 在干净任务上 0/4 完成。原稿还明确无独立 held-out test，外部框架以 3 次运行中最佳验证分数
作排名。以上是“既有报告”，不能作为新增独立确认。正式实验不得在不核查代码的情况下假定旧版具备持
续阶段语义核查、真实 fit 绑定或字段级定向修复。
四项工作相互独立：A 负责真实执行与局部因果；B 负责故障覆盖、合法变化和信息边界；C 负责端到端交付和
资源收益；D 负责自然漂移和封存迁移。各组从同一个 base_commit 克隆，在自己的 lane_X 工作区运行；任何人
不能把另一位成员的新代码、数据或报告作为开工依赖。最终汇合使用共同任务 ID、案例语义族、基线 ID 和日
志格式，而非先后等待。
2. 数据集是“共享基础任务，不共享全部实验案例”
DEV：8 个原稿任务，允许开发与调试。 从原稿第 5 页任务清单建立只读注册条目：Spaceship Titanic、
Titanic、Bank Churn、Academic Success、Ghouls、Obesity Risk、Steel Plate Defect、House Prices。它们已经用于
旧稿，不得包装成全新 held-out 验证。Steel Plate Defect 在新协议里是否可作为真正多标签任务，须先根据标
签共现结构和原始任务说明核验；若不符合，则保持原稿复现用途并修正分类，不得为凑类目强行贴标签。
CONF：12 个新增基础任务，四类各 3 个。 类目编号固定为 CONF-BIN-01..03（二分类）、CONF-MC-01..03（多分
类）、CONF-ML-01..03（真正多标签）、CONF-REG-01..03（回归）。A/B/C 对这 12 个任务使用相同的原始来源、
文件版本、独立评估真值、数据划分、评分规范与 oracle 单元测试，允许各 lane 独立开发 adapter，但须通
过共同判分一致性单测；各自独立生成本工作流所需的故障/合法案例。D 的自然漂移实验也使用这 12 个未被
注入故障的基础任务。相同任务产生多个变异或 seed，不能当成多个独立数据集。
TRANSFER：另 4 个新增基础任务，四类各 1 个。 编号 TR-BIN-01、TR-MC-01、TR-ML-01、TR-REG-01。仅 D 在冻结
方法和迁移模板后开封；A/B/C 不得查看具体任务、封存故障或 D 的测试结果，也不得用它们修改规则。这 4
个任务在统计上也只代表 4 个独立基础任务，不能用数百次重复宣称广泛泛化。
RTS-TAI-PARALLEL-v4.0 · 第 1 页
<!-- page 2 -->
目前 12+4 的具体数据集名称、版本、许可、下载地址和哈希在先前稿件中未完成最终签字，本文不虚构它们已
冻结。 项目负责人须先完成数据注册与独立真值审查，四人才能启动正式 CONF／TRANSFER；四人可在此之
前并行完成 DEV 代码和 pilot。冻结之前，16 个任务必须逐一落实以下编号，禁止让不同成员自行给同一个编
号分配不同数据：
CONF-BIN-01 [具体数据待冻结]   CONF-BIN-02 [待冻结]   CONF-BIN-03 [待冻结]
CONF-MC-01  [具体数据待冻结]   CONF-MC-02  [待冻结]   CONF-MC-03  [待冻结]
CONF-ML-01  [具体数据待冻结]   CONF-ML-02  [待冻结]   CONF-ML-03  [待冻结]
CONF-REG-01 [具体数据待冻结]   CONF-REG-02 [待冻结]   CONF-REG-03 [待冻结]
TR-BIN-01   [封存名称]         TR-MC-01    [封存名称]
TR-ML-01    [封存名称]         TR-REG-01   [封存名称]
访问权限明确分成两份： task_registry_public.csv 对四人公开 DEV+CONF 的完整来源/真值/拆分（oracle 真值
仍只能离线评估器读取），对于 TRANSFER 只公开 TR_id、任务族、sealed_manifest_sha256；
transfer_registry_sealed.csv 保存 TR 的真正名称、来源、私有测试与故障细节，仅由协调人保管，D 在代码和
模板冻结且签署开封记录后获得。A/B/C 直到确认论文主实验结束也不得查看 sealed 表。四份任务书和四份最
终包共享同一个公开注册表 SHA，迁移密封登记只以 commitment SHA 校验，不要求全员提前读取它。数据
选择遵循：来源可访问且许可允许研究；目标和指标有独立权威说明；可建立与训练隔离的私有测试；四类真
实成立；没有与 DEV/彼此的同源任务重复；样本规模及可运行成本满足冻结预算。若同一原始来源拆成多个任
务，source_group_id 相同，并按来源进行独立性审查。
完整注册表在有相应访问权限的视图中 每行一个基础任务，至少包含：task_id, split_tier, source_group_id,
source_name, source_url, license, original_version, source_sha256, task_family, target_columns,
target_semantics, primary_metric, metric_direction, id_columns, output_columns, output_mapping, valid_aliases,
data_split_policy, split_seed, split_hash, oracle_version, applicability_notes, owner_signed_at。缺关键字段不
得从 DEV 晋升为正式任务。原始数据若禁止再分发，仅保存合法的下载指令、哈希和固定索引，而不是擅自打
包。
3. 数据划分、隐藏真值与统一的公开信息边界
同一个 CONF/TRANSFER 任务在所有适用方法和成员间必须使用相同 split_id、相同样本 ID 集与相同私有测试
标签。优先采用经过确认的官方独立划分；否则在正式运行前固定约 60% fit / 20% validation / 20% private
test。分组数据按 group，时序数据按时间，多标签数据用适当的多标签划分，避免同对象或未来信息泄漏。训
练内交叉验证只能使用 fit 部分；validation 用于事先定义的模型选择；private test 只能在离线评价器中读取。
必须存储索引清单与哈希。
项目共同采用数据划分起始随机数 1729；实际拆分实现、库版本和每个任务的 split_hash 应成为冻结文件的一部
分。开发随机块 1101,1102，正式确认随机块 2101,2102,2103，迁移随机块 3101,3102,3103。相同 seed 只是配对设
计；云端 LLM 即便 seed 相同也可能不确定，必须记录真实模型 ID、参数、响应和操作时间（遵守服务使用条
款及脱敏要求）。运行顺序在每个随机块内随机打散并记录，避免全部方法总按固定顺序运行。
私有 C_gold 是评估器持有的真实任务契约，不能传给 agent；C_read 是 agent 入口实际解析出的契约；C_used 是
真实训练/预测动作消费的语义契约。必须分别存储。oracle_gold/、private_test_labels/ 和 agent 工作目录须隔
离；任何通过 prompt、工具上下文、日志提示、缓存或模型选择把隐藏真值提供给受测 agent 的案例均视为
污染，主结果无效。若任务可见材料本身不足以唯一识别目标，应标记“需要澄清/不可辨识”，而非硬把隐藏
答案当作 agent 应当猜中的真值。
信息可辨识层级固定： I-A＝从可见材料可判定唯一任务；I-B＝信息冲突、需要权威证据/澄清；I-C＝相同可见
材料与两个不同真实意图兼容、原理上不可唯一判定。I-C 不能纳入“可检测故障召回”的分母；其适当行为是
谨慎澄清/停止，而不是读取评估器私有答案。合法别名、合法类别映射及语义保持变换必须作为独立的干净负
例。
RTS-TAI-PARALLEL-v4.0 · 第 2 页
<!-- page 3 -->
4. 案例及故障模板固定，允许成员独立实现
共享案例语义编码如下，定义的是要测试的现象，而非全组共享同一个故障脚本。具体变异参数、注入实现和
数据对象由各组独立实现并预注册。
C00：原始干净任务。
B01：输出列有明确合法别名/映射；B02：不改变对象的列顺序变化；B03：明确指定且可逆的类别顺序/目
标变换；B04：无关提示/特征的措辞变化。这四类为合法变化，不应因字面不一致而被误拦。
F01：训练目标缺失；F02：目标不应存在却泄漏到供预测的数据；F03：结构合法但目标提示误导；F04：输
出集合与真实任务不一致；F05：入口后真实训练目标漂移；F06：任务类型漂移；F07：主指标/优化方向漂
移；F08：类别概率顺序或逆变换错位；F09：ID 与预测角色混淆；F10：ID 集合及输出尺寸不变、但预测与
ID 行配对被置换。
P01：目标提示与输出目标的复合冲突；P02：中间阶段漂移加输出语义漂移，必须检查全部真根因。
A01：显性证据冲突且无法权威决定；A02：相同可见证据支持两种不同目标意图，属于不可辨识边界。
I01：dtype/值域等完整性破坏；I02：必需特征/训练测试模式破坏。它们是扩展 I/O 范围，需与核心目标/
提交语义分层报告。
每个正式 case_manifest 至少记录：case_id, task_id, template_id, condition, injection_stage, base_sha256,
changed_paths, mutation_sha256, visible_evidence_hash, C_gold_version, truth_status, information_level,
recoverable, expected_safe_action, target_fault_fields, expected_detection_stage, applicability,
pre_registered_exclusion_reason。故障必须能够通过独立脚本验证真的生效；合法变化须有语义等价证明。一个
案例因不适用而放弃，必须在看到模型结果前标 N/A + 原因；不得凭方法表现再挑案例。F02 人工测试泄漏不得
把私有测试真实标签放入 agent 可见环境。
5. 四人共用的版本与对照方法字典
以下 ID 和定义只能在全体签字后版本化调整；不可用同一个方法 ID 代指两个不同实现。
F0_ORIGINAL：从原稿真实代码 SHA 复现的完整 RTS-MA-DS，保留它原有的契约核查、reviewer 和默认
retry；只读 hook 通过行为等价 smoke 才可称为 F0。若旧代码不可取得或重现，标 not_reproducible，不
得用自行实现版冒称原稿。
F1_EXTENDED：在 F0 之上实际增加的新能力，例如真实 fit 绑定、后续执行检查、字段级定向修复、I/O 合
同；每项要独立列出开关和提交 SHA，只能作为新版本研究，不能把其结果写成原论文旧系统成绩。F1
是可选增强，不是四人的硬依赖。
M0_NO_SEMANTIC_ENFORCEMENT：reader、必要状态更新、正常路由与安全/资源限制不变，只取消研究性的任
务语义阻断；关门控不允许把系统弄崩。
M1_STRUCTURAL_STRONG：字段存在性、类型、shape、ID 合规等强结构规则；不得暗读真实意图。
M2_INTAKE_ONLY_STRONG：与对应 Full 使用相同入口 reader、初始语义规则和冻结契约，只在入口实施语义
检查，后续不做该语义持续核查；这是证明“多阶段增量”的核心强基线。
M3_SHADOW：持续计算相同 reviewer 检查并私下记判决，但其判决不改变执行、不额外透露诊断给 agent。
M4_END_ONLY：保留真实动作轨迹，在末端按事先声明的规则检查，可按相同预算执行末端补救，用于检验
提前检查的时机价值。
M5_BLOCK_ONLY：相关阶段执行同样的判决约束，发现违规即阻断，不提供定向修复，以区分“规避错误”
和“完成任务”。
G_GENERIC_RETRY：与拟比较的定向修复方法匹配次数、时间和模型调用预算，但没有字段级诊断信息；仅
在存在实际定向修复实现时用于恢复归因。
主验证优先顺序： 确认“原稿方法”时主对比 F0_ORIGINAL vs M2_INTAKE_ONLY_STRONG；检验判决是否真实改变动
作时用 F0 vs M3；若 F1 才具备运行中语义约束或定向修复，则 F1 vs M2/G 的结果是新版方法的贡献，不得反写
为 F0。若代码检查证明 F0 与 M2 的入口后检查实际上相同，必须记录这一定义上的重合，不能制造虚假差异。
· 
· 
· 
· 
· 
· 
· 
· 
· 
· 
· 
· 
· 
· 
· 
RTS-TAI-PARALLEL-v4.0 · 第 3 页
<!-- page 4 -->
同一可比较对照的 agent 可见包、LLM/搜索后端、prompt 主体、预算、基础训练器、方法初始化、split 和
seed block 应尽可能相同；有不可能消除的框架固有差异时标为系统级比较，不能把差异归因于单个 gate。安
全沙箱从不关闭。所有补丁必须先通过 version_diff.csv 和只读 hook 行为等价检验。
6. 预算、随机化、运行停止和重跑规则
保留原稿每阶段最多 3 次重试这一已报告设定，但明确 retries 与 attempts 的计数语义，并在代码中加断言。任
务级墙钟时间、模型调用/Token 上限、CPU/GPU/并发数、预算不等价时如何归一化在冻结前填写到 
frozen_config.yaml；先允许用 DEV pilot 测量耗时，不允许先看 CONF 结果再调整预算。强对照必须在同一剩余
预算和同等可见证据下比较；字段级修复与普通重试的次数上限一致。
一个随机块是相同任务、案例、数据拆分、配置的配对标识，不代表两个独立数据集。原始 LLM 输出或行为变
化必须落日志。若调用 API 失败或 GPU 不可用，先保留原运行的 S8 与错误日志；是否可重试、重试次数和重试
后结果如何合并按预冻结规则执行，不能重跑到得到有利结果为止。不因中期效果不显著而随意加 seed，也不
因结果好就提前停止确认。预设安全故障（例如泄露测试标签）可立即中止涉及的批次，并将受污染案例标识
为不可用于确认。
默认各角色共用一个小型 DEV smoke／pilot 任务池：四类任务各 1 个，两个开发随机块；每位负责人依自己方
法做必要 smoke，而非要求所有成员重复完全相同的大型 pilot。预实验只校验可执行性、案例真值、成本估计
与观测完整性；不会因为 F0 在 pilot 中不优于基线而判 pilot 失败。
7. 唯一的八类最终结局与计量口径
每次实际启动的运行必须有且仅有一个互斥终态（没有启动的 N/A 不属于 S1–S8）：
S1 DIRECT_CORRECT：未发生真实修复，训练对象/指标/行对应与真值一致，输出有效。
S2 REPAIRED_CORRECT：至少发生一次实际修复，最终语义正确且输出有效。
S3 JUSTIFIED_STOP_OR_CLARIFY：依据可见信息确实不能安全继续，合理拒绝或澄清。
S4 UNNECESSARY_ABORT：原任务可解却被错误拒绝或合法变换被错误拦截。
S5 SILENT_SEMANTIC_ERROR：生成看似有效的预测文件，实际目标、指标、类别、变换或 ID→预测 关系不符
真实任务。
S6 INVALID_ARTIFACT：提交格式/列/行/数值等明确交付条件不满足。
S7 VISIBLE_EXECUTION_FAILURE：工具或代码可见异常，未发生 S5 语义有效外观的错误交付。
S8 BUDGET_OR_INFRA_FAILURE：预算终止或 API/硬件中断，必须记录 budget 或 infrastructure 子原因。
优先处理同一次运行中的混合事件：若已生成并正式交付有效外观但语义错误的文件，主终态为 S5；若没有有
效交付而崩溃则 S7；资源故障阻断判定时 S8；其他过程中出现的错误还可写在事件序列，不能偷偷出现两个终
态。模棱两可的案例须由离线 oracle 裁定并留存异议记录。
主指标全部采用实际启动运行数作分母，另行给出所有计划格及 N/A：correct_delivery=(S1+S2)/N_started；
silent_error=S5/N_started；unnecessary_abort=S4/N_started；justified_stop=S3/N_started。故障检测的 TP/FN 分母
只包含事前标为可辨识且适用的真故障；合法误拦 FP/TN 分母只包含事前确认的合法场景；根因定位要另报 
exact_match、部分匹配和无定位。正确停止不等于成功交付；发现“早停”不等于发现正确字段。
预测指标只在语义正确且提交有效的运行中报告，再同时给出“以全部启动为分母的正确任务交付率”；
AUC、macro-F1、RMSE 不混合平均。资源报告：真实墙钟秒、错误目标 fit 训练秒、所有 API 调用数、确有
可靠采集时的输入/输出 tokens 和收费；每次正确交付成本 = 该方法全部启动运行的总成本 / (S1+S2 次数)，成功次
数为 0 时记 undefined，不能记 0。提前停止节省了错误训练时间也不能单独证明总体效率更好。
8. 统一统计计划及“有利创新”的预先定义
主统计单位是独立基础任务/独立数据来源，主配对键至少为 (task_id, case_family, seed_block, host_id,
model_id)；因果同前缀实验还必须有相同分叉前快照与候选动作哈希。对每任务先汇总配对差，再给任务级均
值、任务聚类 95% 区间、逐任务方向和留一任务敏感性；若多个任务来自同一源，需要更高层级 
1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
RTS-TAI-PARALLEL-v4.0 · 第 4 页
<!-- page 5 -->
source_group_id 聚类。少量任务时区间本身也不稳，必须诚实表述。二项零事件要有适当上界，不能
拿“0/24”宣称真实风险为零。
主方向预先定为 F0 相对 M2 的静默语义错误率减少，同时检查是否增加 S4、是否损害 S1+S2，以及成本。若 F0 逻
辑等同 M2，则该主张无法由旧方法获得支持；只有明确新增机制并重新冻结研究计划，才能以 F1 建立新版主
张。A 的 F0 vs M3 对动作层有效性是必要但不充分证据；C 的 F0 vs M2 对最终交付是核心；B 与 D 分别提供判定
边界和外部有效性。
建议在 DEV 上预设两个研究实用性容忍参考值：合法/干净条件的正确交付率相对强基线损失不超过 5 个百分
点，每正确交付成本相对增加不超过 20%。它们须经团队签字方可生效，不是 TAI 官方门槛，也不意味着 12 个
任务足以作严格非劣效性证明。若置信区间不足以排除超过容忍界限的损失，应写“未排除有害权衡”，不能
拿点估计替代非劣效检验。
四种结论必须区分：QC_PASS＝实验技术上合格；RESULT_USABLE＝对预先问题有可信、可复核的估计（包括负结
果）；CLAIM_SUPPORTED_STRONG＝预设主对比呈有利方向且任务级 95% 区间与容忍权衡允许强措辞；
CLAIM_DIRECTIONAL / NOT_SUPPORTED / INDETERMINATE＝仅趋势、结果反向或证据不足。质量门控不会被设计
成“Full 必须好于阈值才算过”。 仅统计效果方向有利但区间跨 0，属于可报告的初步信号，不是统计确证。
论文需报告所有预定比较，多重确认检验须在冻结计划中定义调整方案。
9. 通用科研实验质量门控：G0–G6
这些门控是实验是否值得写入论文的资格条件，不是算法自己的阻断规则；四人都要在 quality_gates.csv 中逐
实验填写 PASS/FAIL/NA、证据文件、确认者和时间。
G0 协议签字：FREEZE_SIGNOFF.yaml 存在，CONF 12 个任务的完整公开登记、TRANSFER 4 个任务的封存登
记哈希、每项任务的拆分、base SHA、方法版本、预算、主指标、确认比较、seed、排除/重跑政策冻
结；未签字只能做 DEV，不能声称 CONF 完成。
G1 真值和案例：所有进入主分析的任务目标、适用性、合法等价或故障生效经独立单元测试验证；隐藏标
签/答案与 agent 隔离；每个变更可回溯到唯一变异器及 hash；任何有污染的案例隔离，不用于主分析。
G2 对照可运行：每个本组实际用到的方法通过 DEV smoke；至少两个干净 DEV 任务下基础工作流能正常到
达有效终态。特别禁止把“关闭 reviewer 导致接口断裂”当作方法改善；需要真实阶段动作的实验必须
证明 instrument/hook 工作。
G3 运行账本闭合：每个计划格记 completed/N/A/not_started/infra_failed 等状态；每个实际启动必须落 S1–
S8 之一、唯一 run_id 与原始日志；100% 启动运行可追溯。关键动作缺失只能单列“该动作指标不可
判”，不能静默把该运行从主分母删掉。
G4 比较与污染检查：同一配对的输入、split、seed、可见证据、预算和版本严格匹配或列出不可比较原
因；同前缀因果实验分叉前状态和候选动作 hash 必须相同；private test 无泄漏；差异无法归因门控时不
得称为局部因果证明。
G5 统计与可再现：完整计划格或明确“部分矩阵”；可用冻结环境从原始日志重建逐次、逐任务结果及每
张表图；人工核对至少 2 个有利和 2 个不利/失败记录（若没有有利例，核对全部失败/空事件），数值分
子/分母一致。禁用挑最好 seed、偷偷重跑、事后换类别。
G6 主张匹配：完成每个实验的 QC / USABLE / SUPPORT 三份独立判定，明确独立统计单位、置信区间、反例
与适用范围；不因结果不利判实验失败，也不把故障注入比例当自然发生率。
在某个实验的 G0–G5 实施条件失败时，标 QC_FAIL 并说明污染范围；若只是效果不符合预期但技术条件通过，应
标 QC_PASS, RESULT_USABLE, CLAIM_NOT_SUPPORTED。若完整矩阵未跑完，只能写“部分结果”，不能暗中用预定总
数作分母。代码修复须另记版本、新的实验批次与首次发现日期，不能覆盖原日志。
10. 统一的计划规模、避免重复计算和并行协调
四人共享基础任务，不共享方法成绩，按各自工作负载独立执行：A 计划 144 个相同前缀、最多 720 条干预续
跑和 72 条入口检查分支；B 在 12 个 CONF 上最多 252 个语义案例，最多 1764 次轻量判定；C 计划
· 
· 
· 
· 
· 
· 
· 
RTS-TAI-PARALLEL-v4.0 · 第 5 页
<!-- page 6 -->
12×5×4×3＝720 次端到端完整运行；D 的自然观察 12×2×3＝72 次、迁移 4×2×2×3×3×3＝432 次。上
述数字是计划上限/参考量，不是当前完成数、也不是期刊硬规定；各自的 N/A 和不支持情况需依冻结协议报
告。A 的续跑不是 720 个独立基础任务；B 的“判定调用”不是 1764 次完整训练；D 的 432 次仍只涉及 4 个封
存基础任务。
开工分工：项目协调人（可由作者本人担任）唯一负责发布冻结的只读数据/协议快照、封存目录权限及合并后
的事实核查，不需要替任一成员写代码或跑实验；A/B/C/D 各自配置运行器、变异器、离线 oracle、报告程
序。成员可以提前在 DEV 做小规模独立开发；在 CONF/TRANSFER 的相应门控满足后直接运行自己路线。个人
结果不得通过共享 PR 改动其他人的实验结果；真正需要的共享接口应在本人的 adapter 中完成并记录偏差。
每周只同步“协议问题、复现阻断和资源占用”，不讨论 CONF 中途哪种方法赢、哪些任务对 Full 有利。D 的
TRANSFER 目录及模板 hash 要封存到迁移冻结节点。跨路线最后用同一 task_id 与语义故障族建立证据索引，
不能假定 A 与 C 在同一 task 上的不同 run 是一个实际因果轨迹。
11. 四人共用的开工签字清单（需要在正式运行前真实填写）
原始 RTS 代码仓库地址、base_commit、有无旧故障包、各方法适配实现责任人：[待填]。
CONF 12 个任务的正式名称、来源、许可证、SHA、目标、指标及拆分 hash：[待填，见
task_registry_public.csv]；TRANSFER 4 个任务由协调人对 transfer_registry_sealed.csv 完整签字，四人
共同记录其 seal SHA，D 按规定开封。
原稿 F0 对应实现开关、M2 是否存在与 F0 是否行为不同、F1 是否计划新增：[待填]。
每次完整运行的墙钟/LLM 额度、机器配置、并发数、API 模型 ID、版本、温度、提供商限制：[待填]。
稳健性分层、确认主效应、若启用 5pp/20% 容忍界限的批准签字：[待填]。
公共文件全套 SHA256、签署时间、签署人、封存任务解封时间/责任人：[待填]。
未填写不是“允许各自决定”，而是尚未具备正式 CONF/TRANSFER 研究的 G0 资格。 每个人收到 PDF 后可
以在 DEV 开工，但必须以签字快照替换其结果包中的所有 [待填]。
第二部分　负责人 A 专项：真实执行与局部因果作用
A 的职责、独立性与需要回答的主要科学问题
A 必须独立验证：原稿所谓冻结 target_spec 是否被真实 fit/predict/serialization 消费；reviewer 决策是否实际
改变执行路径；在同一个错误动作即将执行的前缀上，放行、阻断、普通重试、定向修复和末端检查各自引起
什么差异。A 不等 B 给故障包，也不依赖 C 的评测脚本；自己根据共同模板定义四种主要上下文与离线真值。
如果原稿 F0 仅有入口检查而没有持续动作绑定，须准确报告，而不能把 F1 补丁写成 F0 的实验成果。
A 的主要确认主张分成三层：A-C1 实际 gate 能改变错误动作（对 M3）；A-C2 若存在持续核查，则相对 M2 可防
止入口之后的漂移；A-C3 若存在定向恢复能力，则能比仅阻断（M5）或一般重试（G）保留更多正确交付。三
层只能依据各自具有的真实实现宣称。
A0. 旧结果核对、无门控可运行性与代码语义审计
实验设计。 在八个 DEV 原稿任务上核对旧版 Full 与当前代码的一致性。可复现旧配置时，干净任务 8×3=24
次；原稿 16 个故障包逐一做对应回放。再对 M0 与 M3 做不少于 2 个干净任务的完整 smoke，并在 4 个任务族
代表上分析 4 个探针：目标缺失、错误目标提示、合法输出别名、入口后目标替换。
具体步骤。 (1) 固定源代码 SHA 与旧 config；(2) 搜索每个 reviewer 的输入、输出、状态写入和路由依赖；(3)
把“计算判决”和“执行判决”的代码路径分开；(4) 禁止删除 reviewer 所提供的必要状态/回调；(5) 收集
M0、M3、F0 的实际完成轨迹、异常栈与每阶段时间；(6) 对比原稿 Table 3 中关 gate 导致干净任务全部停止的
原因，如果源码不足，明确“原因不可核验”，不填猜测结论。重点检查旧 Table 5 的缺训练目标 2/3 拦截与正
文规则是否冲突，以及原稿不同时长统计的样本来源。
1. 
2. 
3. 
4. 
5. 
6. 
RTS-TAI-PARALLEL-v4.0 · 第 6 页
<!-- page 7 -->
A0 质量门控。 必须拥有可识别的原稿版本或明确不可复现说明；F0、M0、M3 的实现差异可以按函数定位；最
少 2 个 DEV 干净任务中 M0、M3 在不依赖语义 gate 的情况下走完整流程；四探针至少有正确哈希和 expected
outcome。若 M0 因缺少正常状态字段而失败，立即 A0-QC_FAIL，此消融不进入机制结论。
A0 结果可用。 能回答旧数是否重现、差别是代码版本还是实验配置、无门控是否保留必需流程。发现旧消融不
可解释也是可靠负面审计结果。
A0 对创新有利的条件。 在真实保持执行图完整的前提下，F0 的语义阻断确实使已注册错误探针改变下一动
作；仅看到崩溃减少或平均耗时减少，不足以支持语义约束的因果价值。
A0 交付。 A0_old_vs_current.csv、A0_method_diff.md、A0_smoke.csv、reviewer 状态/路由调用图的纯文字事件列
表、用于论文的“旧结果和修正消融”对照表。
A1. C_gold、C_read 与真实 fit/output 的绑定核查
实验设计。 从四种任务族各取 1 个 DEV 任务，每个设置六类探针：正常完成、合法别名、可修复冲突、不可修
复缺目标、planner 后目标漂移、输出类别/ID 配对错位。每个探针分别走 F0 和 M3，默认 4×6×2=48 条开发
轨迹；若真实调用不可拦截，记录 observability gap，不用假造 actual_fit_target。
实施步骤。 (1) 用只读 hook 在 reader、reviewer、planner、实际 fit 调用、predict 和最终文件写入点写 
events.jsonl；(2) 从实际张量源、目标列名及与行顺序绑定的 digest 核对训练目标，不信任 LLM 文本说明；(3)
对输出记录 class_order、inverse_transform、每一行的 ID 映射摘要；(4) 分开标注“reader 最初读
错”“reader 正确但下一阶段改错”“状态正确但最终动作实际做错”；(5) 对不可能抓到的动作事件，报告覆
盖比例和无法证明的边界。
A1 质量门控。 48 个计划格各有运行或预注册 N/A；每条声称“真实执行一致/不一致”的轨迹必须有 C_read、
gate 判决及实际 fit/predict 事件，事件顺序与哈希一致；对输入探针的操作生效应由额外断言证实。若只拿
agent 回答来断言真实 fit，A1-QC_FAIL。
A1 结果可用。 能分别汇总初始契约错误、阶段漂移、实际消费脱钩、输出错配四种失败分母及各自来源事件，
即使 F0 表现差也可用。
A1 创新支持判断。 只有 F0 所批准的契约能稳定约束真实训练和序列化（或者显式拦截漂移），才支持执行真
实性。若原稿只保证共享字段被冻结、却允许真实模型使用别的 y，则原稿执行真实性主张不成立；若 F1 解决
了，要独立报告 F1 新贡献。
A1 论文产物。 T-A1: 真值/入口/真实 fit/输出逐任务一致性表；F-A1: 来自真实 run_id 的动作链轨迹图；附至少一个
正例和一个不符合机制预期的逐事件实例。
A2. 同一候选动作下的放行—阻断—恢复配对干预（主实验）
正式配置。 12 个 CONF 任务 × 4 类上下文 × 3 个确认随机块 = 144 个独立快照前缀。四类上下文：(a) 可恢复
的入口目标/提交冲突；(b) 不可恢复的目标缺失；(c) 入口正确但 planner 后 fit 标签改变；(d) 预测阶段 ID/类
别角色错位。每个前缀最多执行 M3_SHADOW、M5_BLOCK_ONLY、G_GENERIC_RETRY、F0_ORIGINAL、M4_END_ONLY 五条后续
分支，最多 720 个续跑。G 对没有定向诊断可比性的分支可在预注册时标 N/A；F1 若新增，必须单独增加计划
格。
实施步骤。 (1) 依共同任务注册，A 自己制作变异器，确保每次只在声明节点注入，保存前后 hash；(2) 执行到
gate 之前并冻结 state_hash、candidate_hash、visible_evidence_hash、remaining_budget、rng_state；(3) 复制五个
隔离工作区并从完全相同的候选动作继续；(4) 只切换 gate 阻断权、末端位置或重试诊断模式；(5) 逐条记录
next action、实际错误 fit 是否发生、错误文件是否交付、S1–S8 终态、调用与成本；(6) 对没有严格共同前缀
的配对，标为普通随机块比较、从主因果估计剔出但保留运行记录。
主要比较。 F0-M3＝判决执行权对错误动作/静默错误的影响；F0-M5＝允许原稿默认 retry 后是否保留更多交付；
F0-M4＝早期 gate 相对末端核查减少的错误训练是否抵消重试开销；若 F1 实现字段级修复，才计算 F1-G。每个
主张既报阻断率也报 S4（误停止）、S5（静默错）与 S1+S2（正确交付），不能只报“停了多少”。
RTS-TAI-PARALLEL-v4.0 · 第 7 页
<!-- page 8 -->
A2 质量门控。 144 个前缀都有状态且可归档；每个主因果配对前缀的 4 个 hash/预算核验一致，所声明的 gate
策略确实不同；所有实际启动续跑均有互斥终态，存在错误动作的场景均有真实 fit/交付事件；若注入不是唯一
差异或分叉时随机状态失配，剔出主因果比较并说明样本减损原因。CONF 初始版本和修复后的版本不能混为
一个确认批次。
A2 结果可用。 输出按基础任务和四个场景计算的“是否执行错误动作、静默错误率、正确交付率、误拦率、错
误训练时间、总成本”配对差与任务聚类区间，且可定位每一对运行的共用候选动作。
A2 对创新有利的结果。 F0 相比 M3 实际减少了错误执行/静默交付，主要效果任务级区间有利，且没有以大量
S4 换取表面上 S5 降低；若原稿没有真实阶段约束、只对入口故障有益，则必须缩小创新主张；若 F1 有效，另
外给 F1 的效应而不替换 F0。
论文表图。 T-A2: 四类上下文×五种决策的动作与完整终态；F-A2: 逐任务局部效应森林图（含区间）；F-A3: 同一前缀五
种后续动作轨迹（示意线条来自真实事件 ID）。
A3. 强一次性语义检查是否已经足够？
实验设计。 利用 A2 两类前缀——入口冲突和后阶段目标漂移——增加 M2_INTAKE_ONLY_STRONG 分支：
12×2×3=72 条续跑，与相应 F0 的快照配对。入口语义规则、reader、训练器和预算完全相同，唯一研究性差
异是入口之后是否持续验证/阻断。
A3 质量门控。 断言 M2 没有偷偷看到下一阶段的私有故障真值；两方法的入口判决轨迹一致；若代码查明 F0
并无后续语义核查，标为“F0 与 M2 方法作用域重合”，不要构造无法复核的差异。
A3 结果可用。 给出 F0-M2 按入口/后阶段的配对差、区间、对照定义是否成立及其代码证据；零效果也是可用发
现。
A3 有利创新。 对入口之后产生的语义漂移，F0 比 M2 更少执行错误目标或错配输出，并保留正常任务完成；
如果只有 F1 明显改善，只能支持新增持续核查机制，而非原稿已经具有的贡献。
论文产物。 T-A3: 入口一次性检查 vs 原稿完整机制；F-A4: 故障发生位置×两组增量效果；A_claim_ledger.md 列明每个
主张支持、反例、无法观察的边界。
A 的独立结束与最短交付检查
最终提交 lane_A_final.zip、lane_A_evidence_report.pdf；至少含 A0–A3 全部计划格、原始分叉快照与事件、独
立变异器、有效的关 gate 实现、4 张主表图、F0/M2 源码比较及所有负面案例。结论首页分别回答：F0 的行为
约束真实存在吗？错误动作实际改变了吗？与强 M2 有什么增量？是否只是更多拒绝？哪些判断没有可靠观察
条件？
第三部分　四人共同且完全一致的代码、数据、结果与论文交付标准
1. 最终到底交什么给项目负责人？
每位负责人最终提交两件物品：lane_X_final.zip（X 为 A/B/C/D；可复现实验包）和 lane_X_evidence_report.pdf
（建议 4–8 页普通文字＋必要统计图表的结果解释）。实验包中必须同时包含可运行代码、原始证据、全部计
划格索引、统计脚本、结果表图和失败/反例。只给汇总 Excel、截图、几条精简日志或论文措辞，不算通过交
付验收。TXT/MD 源文档可作为辅助，但不能替代机器可读的主记录。
目录及文件名必须遵循共同协议，以便在不等其他人代码的情况下独立打开与重建：
lane_X_final/
  README.md                  # 一页运行说明、研究问题、完成率、主要发现
  BASE_COMMIT.txt            # 原稿 SHA、本分支 SHA、方法哈希
  FREEZE_SIGNOFF.yaml        # 四人完全相同的签字冻结协议快照
  CHANGELOG.md               # F0 / 只读观测 / F1 的版本差异
  SHA256SUMS                 # 案例、配置、日志和表图的哈希索引
  protocol/
    PROTOCOL.md              # 与四份 PDF 的共同协议同版
RTS-TAI-PARALLEL-v4.0 · 第 8 页
<!-- page 9 -->
    task_registry_public.csv # DEV/CONF 完整信息；TR 仅 ID 与密封哈希
    split_registry.csv       # 来源、拆分和隐藏测试 hash
    frozen_config.yaml      # 模型、预算、依赖、seed、温度、硬件
    methods.yaml             # 方法 ID 与真实实现开关
    outcomes.yaml            # S1-S8 与判分优先级
    analysis_plan.md         # 预注册主指标、对比、CI、N/A 规则
    deviations.csv           # 冻结后所有变动/例外及时间
  env/
    environment.lock         # requirements.lock / conda.lock / 镜像摘要
    runtime_info.json        # OS、CPU/GPU、服务模型真实 ID
  src/
    adapters/                # 自己路线实现，独立于其他 lane
    oracle/                  # 离线判分器，运行时与 agent 物理隔离
  cases/
    manifest.csv             # 每个基础任务和案例、适用性、故障真值
    mutation_specs/          # 有版本/哈希的变异器参数与脚本
  scripts/
    smoke.py                 # 最小检查，能自动输出门控结果
    run.py                   # --dry-run / --pilot / --confirm / --resume
    evaluate.py              # 独立 oracle，八种终态与正确任务评分
    analyze.py               # 任务级配对统计、CI 与敏感性
    make_paper_assets.py     # 从原始记录重建全部论文表图
  logs/
    plan_ledger.csv          # 包括未启动/N/A 的全矩阵
    run_index.csv            # 唯一 run_id、路径、SHA、状态
    events/*.jsonl           # 每次运行的真实判决/动作/模型/交付事件
    errors/                  # API/工具/硬件错误、失败重试和回溯
  outputs/
    artifact_index.csv       # 产物 hash、判分、缺失/无效原因
  results/
    run_level.csv            # 一启动一行，失败不得删
    pair_level.csv           # 同任务/同前缀可比性键
    task_level.csv           # 独立任务级效果、区间和方向
    resource_ledger.csv      # 时间、调用、训练浪费、缺失计量
    quality_gates.csv        # G0–G6 + 本 lane 每实验门控
    claims.csv               # QC/USABLE/SUPPORT 的三层结论
  paper/
    tables/*.csv             # 正文/附录每张表的数字及分母
    figures/*.pdf            # 简洁黑白论文图，由脚本生成
    figures/*.png            # 查阅预览
    cases/*.md               # 正例、反例、无效例及事件 ID
    evidence_report.pdf      # 本人成果的可写作结论与局限
  reproduce.sh               # 只使用本目录与合法外部数据来源
task_registry_public.csv 只包含 A/B/C 可见的 DEV/CONF 细节与 TRANSFER 封存哈希；A/B/C 不得提前得到
TRANSFER 名称、私有标签或故障内容。D 的有权限私有副本应在封存开封后单独归档。若有关文件因版权/
API 许可不能入 ZIP，应交非敏感引用索引、合法获取命令、哈希和权限说明，并在 README 中明确是否可完
全公开复现。密钥只能通过环境变量读取，禁止打包 API Key、测试标签明文或匿名投稿受限文稿。
2. 所有 lane 的最少日志字段：必须逐行可连接
plan_ledger.csv：plan_cell_id, lane_id, stage, task_id, source_group_id, condition, template_id, case_id,
method_id, host_id, model_id, seed_block, planned_status, applicability, n_a_reason, freeze_sha。每个计划格都
在运行前生成，执行后更新状态；主论文标明实际启动数、完成数、N/A、未运行数，不可把 N/A 记为失败率分
母中的 0。
run_level.csv：run_id, plan_cell_id, pair_id, task_id, source_group_id, case_id, case_family, method_id,
implementation_sha, split_hash, oracle_version, prompt_sha, config_sha, seed_block, host_id, model_id,
started_at, finished_at, terminal_S, terminal_subreason, semantic_correct, artifact_valid,
actual_fit_target_hash, actual_metric, output_mapping_valid, rejected_valid_case, wall_seconds,
wrong_training_seconds, api_calls, tokens_in_if_observed, tokens_out_if_observed, measured_cost_if_observed,
artifact_sha256, event_trace_path, exception_path, missingness_reason。确实无法观测的字段必须写 NA 及原因，
不能写 0；不能靠 agent 自评来填 semantic_correct。
RTS-TAI-PARALLEL-v4.0 · 第 9 页
<!-- page 10 -->
events.jsonl：每条至少含 event_id, run_id, monotonic_seq, timestamp, phase, actor, event_type, state_hash,
candidate_hash, contract_digest, source_fields, verdict, next_action, actual_fit_target_hash,
artifact_mapping_digest, error_subtype。应能从 reviewer 判决追到真实 fit/predict 或合理停止，且顺序单调。
可复核性最好提供至少一条正向和一条反向的完整轨迹索引。
quality_gates.csv：experiment_id, gate_id, status, checked_count, eligible_count, evidence_path, reason,
checked_by, checked_at。claims.csv：claim_id, experiment_id, primary_contrast, estimand, units, estimate,
ci_low, ci_high, qc_status, result_usable, support_level, counterexample_ids, allowed_paper_wording。任何论文
中的“改善百分比”都应能返回这些记录核验。
3. 代码提交最低验收：不靠作者口述即可运行
./reproduce.sh --smoke 从干净环境加载冻结配置并完成最小测试；需要外部私有数据时明确 DATA_ROOT 与下
载授权，不能偷偷使用作者本机绝对路径。
python scripts/run.py --dry-run 打印实际矩阵、计划数、N/A 和成本预估；--pilot 与 --confirm 分阶段运
行，--resume 能以唯一 run_id 恢复，绝不覆盖已存在的完成日志。
同一个源文件只做一次 SHA256 登记；环境/接口变更带新 implementation_sha，所有无门控、shadow 和
检查变体能够通过配置独立运行，不能修改共享源文件“硬切换”后丢失版本。
python scripts/evaluate.py 离线读取 oracle/私有测试，不依赖 agent 主观自述；对八终态、输出行映射、
误拦、预算失败均有固定单元测试。
python scripts/analyze.py 由全部启动运行生成分母和配对统计；python scripts/make_paper_assets.py 自动
重建每张表/图，并保存图表至运行文件及统计单位的溯源索引。
除必要第三方外，不允许依赖 ../lane_A、../lane_B、../lane_C、../lane_D 等别人的私有目录。若发现不可
移植，提交前调整为自己的 adapter 或经签字公开的共同 package。
以上命令是交付时要求成员真实实现并测试的接口规范，不是声称当前已有可以直接执行的脚本。
4. 什么程度的记录足以用来撰写 TAI 论文？
Methods: 可以精确定义 F0 的契约、检查频率、路由权、失败语义和可观察动作；若实际实现不同于论文伪代
码，能用代码差异与完整版本审计说明。要有 F0 对照、强 M2 的算法/接口差别，不用代理文字叙述代替源码。
Experimental setup: 每个任务有来源、目标、私有测试隔离证明、分层和独立性；每个故障有代码、可恢复
性、正确响应/字段与可见信息；每个方法有真实预算和输入；所有不支持格子、技术故障和除外规则可公开复
核。
Main results: 每一个主效应都有完整分母、独立任务数、置信区间、逐任务差、比较方法、三类完整终态风险
（静默错、正确交付、误拦），不采用 best-of-three 或只统计“能写出文件”的系统。若 12 个任务得不出精确
结论，可以如实报告不确定性，并考虑预先注册的独立扩展，而不是复制 seed 增加名义样本数。
Mechanism evidence: 事件轨迹能从 C_read 到真实 fit/提交映射，再到 gate 决策和分叉后结局；至少一个可
核对的正向案例及一个不支持该主张的反例（若真实不存在正向案例，说明不存在，不得捏造）。同前缀比较
还提供状态与候选动作 hash。
Boundary and transfer: 故障覆盖须报告“合法变体误拦”与不可判定问题；自然漂移须说明注入与非注入区
别；迁移须报告两个宿主、两个模型家族和未见模板在每个单元的真实运行情况、失效组合和精度限制。
Paper assets: 每张图/表有文件源、生成命令、归一化方式、配对层级和引用的原始运行索引；正文不要出现
源文件里无法核验的新数字。给出一份“证据链矩阵”：主张 → lane → 实验 ID → 表图编号 → 最小复现命令
→ 支持/限制/反例。
1. 
2. 
3. 
4. 
5. 
6. 
RTS-TAI-PARALLEL-v4.0 · 第 10 页
<!-- page 11 -->
5. 负责人最终怎样验收四个实验包？
交付合格（PACKAGE_PASS）：压缩包结构、冻结快照、原始日志、失败记录、重现脚本、图表都齐
全；能在清洁环境完成 smoke，并重建本人的主要表格。否则退回补齐，即使图表方向很有利也不算完
成。
结果可用（RESUL T_USABLE）：基础任务独立性、案例真值、对照公平、终态完整、配对估计和不确定
性符合本 lane 的问题；负结果同样可能可用。若数据污染、对照失效、哈希不匹配或主分析没有完整分
母，应标 NOT_USABLE 并说明需重跑哪部分。
创新支持（CLAIM_SUPPORTED）：A 证明动作层因果，B 证明语义边界的额外检测价值，C 证明端到端
静默错误减少并保留正确交付，D 给出自然问题证据/真实封存迁移。四项共同支持有条件的、证据可跟踪
的 TAI 主线；单项漂亮结果不能替代其他环节，负结果可能导致主张收窄或方法升级。
论文最终总结时，还必须给四条证据链之间的一致性审计：相同语义故障在 A 的动作差、B 的定位差、C 的终态
差是否方向兼容；若不兼容说明原因，不要把不同成员不相干的 run 硬拼成一条实际因果样本。是否达到 TAI
的发表水平还取决于方法原创性、相关工作定位及证据规模，门控通过并非期刊接受保证。
· 
· 
· 
RTS-TAI-PARALLEL-v4.0 · 第 11 页```
