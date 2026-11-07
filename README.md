# AI Modeling 思想实验

这个工程用于持续讨论 AI modeling，重点是思想实验、概念澄清和可检验的推论。这里的“modeling”暂不预设单一含义；讨论中先定义所用含义，再推演。

## 从哪里开始

继续当前研究时，先读[第一部分主方案 v1.2](path_control_pretraining_part1_v1.2.md)、[权重空间编程对 B 研究的启示与下一步（2026-11-03）](discussions/2026-11-03-weight-space-programming-and-B-research.md)和[讨论索引](discussions/README.md)。Dot 研究分支已合入 main，v1.2 原文字节保留；其明确修订处优先于历史版本，不以聊天中的不严谨简称替代设计原文。近期优先可观察状态、因果干预与知识增量，效率暂不作为研究硬门槛；单级与多级仍需公平比较。[分支接续记录](discussions/2026-11-02-v1.2分支与权重空间编程接续.md)保留来源与证据边界。

当前训练设计按用户 11 月 6 日进一步明确的口径：先完成 B⁺／B⁻ 详细设计，以 B⁺、B⁻、F 联合训练建立 baseline，再研究参数空间重编程能否加速；此前“先训 B 再冻结”的顺序保留为历史选择或对照。输入 embedding 已训练好是本轮给定条件；用户确认输出向量 label 指底层输出 token 的已有向量，具体资产与损失配方仍待确定。F 和最终展开器从头训练，0.8B 指目标参数规模，不复用 Qwen 0.8B 主干。见[B 双向设计与联合训练基线](discussions/2026-11-06-B双向设计与联合训练基线.md)。第 0 步已实现[固定词边界数据管线与小样例](training/word_boundary_v0/README.md)，沿用 v1.2 §10 的监督及可见性要求；尚未训练模型。

[第 0 步增量 v0.2](training/word_boundary_v0_2/README.md)解释关键夹具，增加 4 条独立现代中英文人工审核候选、有界 FMM 流式参考及真实 tokenizer 盘点。46 项结构测试通过，人工金标仍未接受；FMM 的“研究生｜命”反例保留，不以有界／因果测试代替分词质量。旧数据包与 split 保持原样。

[H-Net 完整 ROCm 参考路径](experiments/hnet_rocm/README.md)已在 8060S 上实测通过 874M 模型的完整 GPU 前向、全参数反向和一次 SGD 步，并完成 CPU 数值对照。它是 PyTorch HIP 替代实现；原 fused 内核与增量推理缓存未移植，不等于本项目 v1.2 模型已训练。专用分支保留旧 CPU 探针及数据包。

[三模型本地词／词组标注 pilot](training/generation_v1_2/local_pilot_2026_11_04/README.md)已下载／校验 Qwen3.6-35B-A3B、Qwen3.8-27B 和 GLM-4.7-Flash Q8_0，8060S llama.cpp Vulkan 各完成同一 100 条，共 499 次尝试。首轮28条重试后硬通过13／20／4；另72条模板变体42／42／0。36份固定抽样及24份模板补充由Codex目测，参数口径、原始失败及六类错误可追溯；硬通过不代表语义合格，均未入训。395仅测试，MI300生产性能待实测，暂缓H-Net fused。

[第一部分主方案 v1.1](path_control_pretraining_part1_v1.1.md)、[2026-10-06 接续说明](HANDOFF.md)和[2026-10-02 接续说明](HANDOFF-2026-10-02.md)保留作历史记录。已有有限规则解析器和词表示压缩诊断；尚无通用语义 B、B⁻、Qwen F 接入或完整多粒度模型训练结果。

[B⁻ 层次调用、Lookback 与训练监督协议 v0.1](B_minus_lookback_supervision_protocol_v0.1.md)细化跨尺度调用及监督隔离，已由 v1.2 整合修订；协议实现仍待验证。

此前会话原文见 [2026-10-02 Transcript（筛选版）](transcript-2026-10-02.md)，已排除指定的两轮无关对话。

本轮关于层深、可逆性、读写分工、预训练与渐进式部件替换的对话原文见 [2026-10-04 Transcript](transcript-2026-10-04.md)。

1. 阅读 [AGENTS.md](AGENTS.md) 和 [讨论索引](discussions/README.md)。
2. 在新讨论开始前，阅读索引指向的最近记录和相关主题记录。
3. 将有持续价值的讨论写成新的记录，并更新索引。

Git 历史保存每次修订。Markdown 文件保存可被后续 agent 直接读取的讨论记忆；聊天记录本身不视为唯一存档。

## MASA 设计基线

本项目由 [MASA 设计](masa.md) 等早期讨论演进而来。当前主方案研究模型内部的 path control：按内容和任务选择计算路径与处理粒度；MASA 是历史启发，不是必须先完成的实现基线。

MASA 与动态递归、跳层和多粒度状态计算之间的范围差距见[设计边界讨论](discussions/2026-09-29-masa-scope-and-dynamic-computation.md)。导入基线提供已有机制与待验证假设；当前路线以本页所指主方案及最新研究接续为准。

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
