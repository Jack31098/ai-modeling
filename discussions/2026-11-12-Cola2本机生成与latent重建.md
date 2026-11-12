# Cola2 本机生成与 latent 重建

问题：在 v1.2 研究中，是否有真正运行的「先预测连续 span 表示，再生成文字」基线？这次验证 ByteDance-Seed [Cola2](https://arxiv.org/abs/2610.31305) 的正式公开 checkpoint。它是独立研究对照，不改变 v1.2 的 B 训练顺序、参考规约监督和本地／MI300 分工，也不恢复此前暂缓的 H-Net fused。

**事实。** 官方 [推理源码](https://github.com/ByteDance-Seed/Cola2/tree/085817168c9d6a718812fbc0075b968fc4cd5f9f) 和 [checkpoint](https://huggingface.co/ByteDance-Seed/Cola2/tree/db876b905cf719b60da33c12a75021ac111f7a67) 已分别核实，不将论文开放当作权重开放。D 盘八个 FP32 分片 35.072 GB，SHA-256 全过。8.768B 参数由实际模型计数，含 4.039B latent DiT 和 4.023B token decoder；另有一个持久的分段偏置标量缓冲。代码许可 Apache-2.0，权重卡没有独立许可证字段。无需 HF 登录，没取数据集或租机器。

**观察。** Windows 8060S/gfx1151、原有 PyTorch ROCm 7.2.1 与 BF16，官方问题提示输出 32 token，7.54 秒，2 轮去噪／解码／回编码，峰值 PyTorch allocated 17.84 GB。先产生「Paris.」，后续出现重复与错误。短句 `The red box holds 17 coins. Keep it by the window.` 压到 9 个 64 维 latent（15→9）。仅给 latent、让时长头预测布局、解码从空前缀开始并用自身输出作为后续前缀，此例还原了所有 15 个 token。已另存带原文前缀的官方重建对照，两种证据不混用。

**与 v1.2 的关系。** Cola2 真正对 latent 采样，再做离散 span 实现；同一次生成内部可看到连续去噪、时长、token 提交和再编码，因而能作为研究 B⁻ 如何落成文字的实验对照。已提交文字用固定布局重新编码，是连续历史与实际文字对齐的一种具体做法；这不等于 v1.2 有权限和结果消费规则的 RETURN／Lookback，不证明多轮新增尺度或外层遗忘有效。34 层 DiT、36 层 decoder 和每次 8 个 latent 的块都不能当作递归栈深度或独立规约层数。发布包展示推理，未交付足够的训练复现材料；不能从 demo 反推可复现训练或 v1.2 的路线优劣。

**反例和未决问题。** 单一短句成功，不能把 9/15 当成通用信息码率或无损保证。下一组最小反例可配对 `17`／`71`、互换角色、增加否定，分别测预测时长重建与已知时长重建，区分信息丢失和时长错误。更重要的闭环对照是固定 prompt、seed，只开关提交后的回编码，测后续文字、实际状态变化和精确信息保真；本轮没有进行这项因果干预。

完整输出、命令、版本锁、CPU／GPU 边界和失败证据见 [Cola2 本机实验](../experiments/cola2/README.md)。
