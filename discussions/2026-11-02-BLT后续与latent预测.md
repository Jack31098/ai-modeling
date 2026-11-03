# BLT 后续与直接 latent 语言建模

日期：2026-11-02（America/Los_Angeles）。状态：一手文献与实现核查、架构推论；本轮没有复现论文或训练模型。

## 问题、术语与范围

用户问 Byte Latent Transformer（BLT）有哪些后续，以及规约既然可训练，为何原团队没有直接用规约后的 latent token 训练 LLM。

必须区分三件事：让大主干以压缩状态为输入；预测下一压缩状态作为生成目标；把自己预测的状态直接作为下一步输入。BLT 做了第一项，不能因其名含 latent 或图中写“predict next patch”就认定另两项也已实现。连续 patch 向量也不自动等同于离散、稳定或语义规范化的符号。

本文按本地日期截至 2026-11-02 的可读来源整理。直接续作、独立相关进展与同期替代路线分别标记，不把所有引用 BLT 的论文当作原团队后续。

## BLT 已经做了什么

[原论文 §2–3](https://arxiv.org/html/2412.09871v1)与[正式 ACL 版本](https://aclanthology.org/2025.acl-long.453/)给出的路径是：局部编码器形成连续 patch 表示，大 global Transformer 处理该短序列，局部解码器产生下一字节的分布。主任务是 next-byte cross entropy，梯度训练连续编码、主干与解码模块。

关键结构细节是：local decoder 还接收 local encoder 的逐字节 hidden states（§3.3、式 9），并结合过去字节进行因果预测，不是提前读取目标未来字节。该训练没有要求单个 global patch latent 独自承担完整解码条件，也没有将独立可重构的 codec 作为验收目标；不能据此反推 latent 必然信息不足或不可独立解码。输入 patch 表示与 global 输出的下一段条件状态也不是同一对象。

官方实现可核验：[global forward 直接接收 embeddings](https://github.com/facebookresearch/blt/blob/main/bytelatent/model/latent_transformer.py)、[整体 E/G/D forward](https://github.com/facebookresearch/blt/blob/main/bytelatent/model/blt.py)、[最终 CE 损失](https://github.com/facebookresearch/blt/blob/main/bytelatent/train.py)。原版没有把下一个 encoder latent 当作显式回归目标。

原版按独立小 byte LM 的预测熵设置 patch 边界；主模型连续表示可端到端训练，不等于该熵切分决策也通过同一任务梯度联合优化。实现中的熵计算默认关闭梯度，见[patcher](https://github.com/facebookresearch/blt/blob/main/bytelatent/data/patcher.py)。预测熵相关的边界也没有语义单元最优性的保证。

## 三条最相关的后续

**H-Net，2025-07，独立相关推进。** [论文](https://arxiv.org/html/2507.07955v2)、[作者代码](https://github.com/goombalab/hnet)将 chunk 边界与语言模型联合学习，并允许递归多级组织。它对“分组本身能不能训练”提供直接证据，但仍从输出字节任务训练整体，未建立稳定的语义码本或纯 next-latent 生成。论文规模边界不能外推为任意大模型已验证。

**Fast BLT，2026-05，含原 BLT 作者的直接续作。** [论文](https://arxiv.org/html/2605.08044v1)针对逐字节生成瓶颈，提出 BLT-D 的块扩散并行解码、BLT-S 的局部解码器草拟加全模型验证，以及 BLT-DV。其 1B/3B 实验的主要效率依据是网络调用次数与估算内存带宽；§7 明确优化实现上的实测吞吐／延迟留待后续。它没有把主损失整体换成下一 encoder latent 的回归。

**Bolmo／byteification，2025-12 首发，2026-10 扩展。** [原论文](https://arxiv.org/abs/2512.15586)、[2026-10-07 扩展全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13686933/)、[Ai2 发布](https://allenai.org/blog/bolmo-nature)保留已有 subword LLM 的 global 主干，先冻结它并训练 byte 接口，再全模型适配。扩展工作包括 Qwen3 8B Base 和 Llama3 8B 的转换。这说明“压缩表示接已有 LLM”已有实作，接口仍须适配；最终也保留字节 CE，不是纯 latent 序列目标。其使用边界监督，不应与 H-Net 的端到端边界学习混同。

## 直接预测 latent 的路线确实存在

**LCM，Meta，2024-12，与 BLT 同期的独立路线。** [Large Concept Models](https://arxiv.org/html/2412.08821v1)冻结 SONAR 句子 codec，直接建模下一句 embedding 的条件分布；比较 MSE、扩散及量化方案，并扩展到 7B。它是“先规约／编码，再在该 latent 空间训练生成模型”的直接先例。论文报告单纯 MSE 较弱，并分析多模态续文、编码空间脆弱性及数字／代码保真等边界；不能因此认定所有 latent LM 都存在相同程度的缺陷。

**CALM，2025-10。** [论文](https://arxiv.org/html/2510.27688v1)先训练将 K 个 token 压到一个连续向量的稳健 autoencoder，再学习连续下一向量分布，采用带噪声的生成头与 energy loss。其报告的高重建率限定于该 codec 测试，不等同于生成任务已无损。

必须保留 CALM §3.3.3 的重要限制：作者发现直接把预测 latent 输入 Transformer 会降低表现；最终实现的反馈是预测向量 → 冻结 AE decoder → K 个离散 token → embedding/MLP 压缩为主干下一步输入。它证明连续生成目标可以训练，但不能被写成不经文字的纯 latent 回灌闭环。LCM 与 CALM 对这两个问题给出了不同的选择。

## 对“为何没有直接采用”的解释边界

未在原 BLT 论文或官方实现说明中找到作者明确报告“完整比较后否决冻结 codec＋latent AR”的依据。因此以下是架构分析，不冒充作者的历史动机。

1. 可学习的压缩计算接口不自动是可独立生成的 codec。若移除 BLT 原解码器的细粒度条件，必须重新验证信息保真；不能只改损失名字就继承原模型全部能力。
2. 连续下一向量需要建模分布。已知前缀可能允许红／蓝两种续文；若示例编码是 +1／−1，单点 MSE 的最优解为 0，未必对应任一合法表达。问题来自点回归的目标，不是连续空间不可用；混合分布、扩散、flow、energy 方法或离散量化均可作为候选。
3. 重建、生成与回灌分别验收。精确编码点能重建，不保证预测偏移后还能解码；预测向量能解码，也不保证它易被下一步主干使用。CALM 的输入消融正是具体证据，不能扩张成普遍不可能性结论。
4. 若选择离散 codebook，可以继续用常规 next-code CE，但需训练量化接口，记录码本容量、保真与实际步数；它是新增的学习型符号系统，并非原 BLT 已经具备的词表。

冻结 codec 后，预测器训练本身不会使固定编码目标自行坍缩。目标漂移或表示坍缩主要是联合训练且约束不足时另需处理的风险，不应笼统用来否定任意连续 latent 模型。

## 与本项目 v1.2 的关系、对照与下一步

v1.2 以参考结构监督动作，让连续状态在真实任务路径上联合适配；这与 BLT/H-Net 在粗粒度主干上学习的思路有共通处。BLT 的边界、细粒度状态通路和输出反馈并不因此成为本项目必须复制的接口，也不能当成多级语义规约、B⁻ call/return 或 latent 树搜已获验证。

可独立比较两类目标：在同一参考切分和压缩率下，连续 E/F/D 联合接受文字／任务损失；或先学并固定 codec，再训练下一 latent 的分布生成模型。MSE 可以作为诊断对照，不预设为默认优胜方法。这种对照不要求同义 latent 相等，也不重新引入旧 hidden 的逐维对齐。

最小验收包括 codec 重建及扰动后解码、角色／否定／精确数字、真正自主多步续写、输出与内部状态的一致性，以及接入已有 F 的适配范围。分别计主干步数、编码／解码与采样成本；不能只看 teacher-forced latent loss 或节点数。

暂定判断：用户提出的直接 latent 语言建模是可研究且已有先例的方向；从“规约可训”到“适合独立生成并长期回灌的规约码”仍需额外证据。对现有项目，H-Net 对应可学边界与递归组织，Bolmo 对应继承已有 F，LCM/CALM 对应生成目标与反馈接口，可分别借鉴。
