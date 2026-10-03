# AI Modeling 思想实验

这个工程用于持续讨论 AI modeling，重点是思想实验、概念澄清和可检验的推论。这里的“modeling”暂不预设单一含义；讨论中先定义所用含义，再推演。

## 从哪里开始

换机继续本轮研究时，先读 [2026-10-02 接续说明](HANDOFF.md)，其中汇总当前进度、最近的概念修正和未决问题。

本次会话的用户消息与助手最终回复原文见 [Transcript（筛选版）](transcript-2026-10-02.md)，已排除指定的两轮无关对话。

1. 阅读 [AGENTS.md](AGENTS.md) 和 [讨论索引](discussions/README.md)。
2. 在新讨论开始前，阅读索引指向的最近记录和相关主题记录。
3. 将有持续价值的讨论写成新的记录，并更新索引。

Git 历史保存每次修订。Markdown 文件保存可被后续 agent 直接读取的讨论记忆；聊天记录本身不视为唯一存档。

## MASA 设计基线

本项目从 [MASA 设计](masa.md) 继续演进，研究不可变内容流与可变思想流、latent 工作记忆及其读写和推理机制。

当前正评估 MASA 与动态递归、跳层和多粒度状态计算之间的范围差距，见[设计边界讨论](discussions/2026-09-29-masa-scope-and-dynamic-computation.md)。导入基线提供已有机制与待验证假设，最终架构范围仍在讨论中。

- 上游：[Jack31098/inf_context_llm](https://github.com/Jack31098/inf_context_llm)。
- 导入文件：[masa.md 固定版本](https://github.com/Jack31098/inf_context_llm/blob/1890b34d6b39fd24dcf5d9afbf04de9ba64f5f27/masa.md)。
- 上游提交：`1890b34d6b39fd24dcf5d9afbf04de9ba64f5f27`。
- 上游文件 blob：`93dc887dc18ca42e8407141c643227d93ebd12fc`。
- 导入日期：2026-09-29。初始内容与上游一致，后续在本项目的 `masa.md` 中独立修改；Git 历史保留导入基线。
- 当前技术判断见 [MASA 评估](discussions/2026-09-29-inf-context-and-masa-review.md)。设计中的能力与性能主张仍需验证；原文的状态标签不代表本项目已完成实现或实验。

## 目录

- `masa.md`：从上游 fork 的 MASA 设计，本项目后续演进的基线。
- `discussions/`：按日期和主题归档的讨论记录。
- `.cursor/rules/`：Codex/Cursor 会话中的常驻规则。
