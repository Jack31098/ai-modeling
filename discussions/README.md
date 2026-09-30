# 讨论索引

本目录是跨会话的讨论记忆。新 agent 从此页找到相关记录，再阅读原文。记录按问题组织，状态随讨论更新。

## 当前状态

- 已导入一段 [Optimal mask 元学习方案分享对话](../share-transcript.md)，原始网页存于 [`share.html`](../share.html)。
- 讨论当前聚焦：接续用户已有的 MASA 双流设计，补全 latent 思想状态到内容输出的直接通道、训练目标及修订一致性。
- 已归档 [World Labs 世界模型分享对话原文](../share-world-labs-transcript.md)，原始网页存于 [`share-world-labs.html`](../share-world-labs.html)；[讨论摘要](2026-09-29-world-labs-to-hierarchical-memory.md)梳理了从路线比较到动态层级计算图、选择性遗忘与主动上下文学习的转折。
- 已完成[层次化记忆 vision 的初步技术评估](2026-09-29-hierarchical-memory-feasibility-roadmap.md)；随后读取用户既有设计，修订了将文本记忆作为项目起点的建议。相关组件有可行性证据，完整架构的优势尚未证实。
- [inf_context_llm 与 MASA 评估](2026-09-29-inf-context-and-masa-review.md)区分语义载荷/索引基础设施与主动双流工作记忆，指出 mask 反馈、attention 蒸馏、KV 修订依赖及 Oracle 因果性等关键缺口，并给出接续实验。
- 尚无本项目实验结果；已核查的论文发现限定于各自设置，具体架构与 roadmap 属于待验证建议。

## 记录

- [2026-09-25：项目起点](2026-09-25-项目起点.md) — 项目方法与初始开放问题。
- [2026-09-25：从 optimal mask 到 latent 工作记忆](2026-09-25-optimal-mask-to-latent-memory.md) — 分享对话的脉络、最近问题和待验证分歧。
- [2026-09-29：从 World Labs 到层次化可写记忆](2026-09-29-world-labs-to-hierarchical-memory.md) — 新分享对话的主要转折、架构假设与失败条件。
- [2026-09-29：层次化记忆的可行性与 roadmap](2026-09-29-hierarchical-memory-feasibility-roadmap.md) — 技术证据、KV 与语义层级等概念修订、候选路线、分阶段验证和停止条件。
- [2026-09-29：inf_context_llm 与 MASA 设计评估](2026-09-29-inf-context-and-masa-review.md) — 修订研究起点，以双流直接读出和可变状态的一致性继续推进。

## 续接方式

开始新话题时，先读与主题相关的记录；讨论后更新该记录或创建新记录，并在这里增加链接与一句摘要。修正旧观点时同步改动其状态描述。
