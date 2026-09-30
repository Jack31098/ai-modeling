# 讨论索引

本目录是跨会话的讨论记忆。新 agent 从此页找到相关记录，再阅读原文。记录按问题组织，状态随讨论更新。

## 当前状态

- 已将 [MASA 设计](../masa.md) fork 到本项目，固定上游版本并保留原文，作为后续设计演进的基线；来源见[项目 README](../README.md#masa-设计基线)。
- 已导入一段 [Optimal mask 元学习方案分享对话](../share-transcript.md)，原始网页存于 [`share.html`](../share.html)。
- 讨论当前聚焦：MASA 与动态递归目标的差距，尤其是训练如何启动。需同时判断可执行路径与可训练性；先验证有用的内部更新算子，再研究动态调度，不能把结构可表达当成训练可行。
- 已归档 [World Labs 世界模型分享对话原文](../share-world-labs-transcript.md)，原始网页存于 [`share-world-labs.html`](../share-world-labs.html)；[讨论摘要](2026-09-29-world-labs-to-hierarchical-memory.md)梳理了从路线比较到动态层级计算图、选择性遗忘与主动上下文学习的转折。
- 已完成[层次化记忆 vision 的初步技术评估](2026-09-29-hierarchical-memory-feasibility-roadmap.md)；随后读取用户既有设计，修订了将文本记忆作为项目起点的建议。相关组件有可行性证据，完整架构的优势尚未证实。
- [inf_context_llm 与 MASA 评估](2026-09-29-inf-context-and-masa-review.md)区分语义载荷/索引基础设施与主动双流工作记忆，指出 mask 反馈、attention 蒸馏、KV 修订依赖及 Oracle 因果性等关键缺口，并给出接续实验。
- [MASA 的范围与动态计算](2026-09-29-masa-scope-and-dynamic-computation.md)进一步区分记忆组织与计算调度：MASA 有独立的局部目标，但现有 A→B→C 路线不会自动产生内部层回路或跳层。上限判断限定于现有执行方式，尚无通用能力上限的证明。
- [动态递归的训练脚手架](2026-09-29-recurrent-training-scaffolds.md)回应训练启动难题：MASA 的语义示范不能直接教授内部路由；受限循环有结构与优化脚手架，但联合自由调度尚无本项目验证。修订上一轮对最小原型可训练性的隐含乐观。
- 尚无本项目实验结果；已核查的论文发现限定于各自设置，具体架构与 roadmap 属于待验证建议。

## 记录

- [2026-09-25：项目起点](2026-09-25-项目起点.md) — 项目方法与初始开放问题。
- [2026-09-25：从 optimal mask 到 latent 工作记忆](2026-09-25-optimal-mask-to-latent-memory.md) — 分享对话的脉络、最近问题和待验证分歧。
- [2026-09-29：从 World Labs 到层次化可写记忆](2026-09-29-world-labs-to-hierarchical-memory.md) — 新分享对话的主要转折、架构假设与失败条件。
- [2026-09-29：层次化记忆的可行性与 roadmap](2026-09-29-hierarchical-memory-feasibility-roadmap.md) — 技术证据、KV 与语义层级等概念修订、候选路线、分阶段验证和停止条件。
- [2026-09-29：inf_context_llm 与 MASA 设计评估](2026-09-29-inf-context-and-masa-review.md) — 修订研究起点，以双流直接读出和可变状态的一致性继续推进。
- [2026-09-29：fork MASA 设计基线](2026-09-29-masa-fork.md) — 导入上游原版设计，确定本项目的接续位置与版本来源。
- [2026-09-29：MASA 的范围与动态计算](2026-09-29-masa-scope-and-dynamic-computation.md) — 独立目标、执行边界与候选总体架构，修订下一步的优先级。
- [2026-09-29：动态递归的训练脚手架](2026-09-29-recurrent-training-scaffolds.md) — 区分语义示范与结构约束，分析算子和控制器的共同启动难题及可证伪的分阶段候选路线。

## 续接方式

开始新话题时，先读与主题相关的记录；讨论后更新该记录或创建新记录，并在这里增加链接与一句摘要。修正旧观点时同步改动其状态描述。
