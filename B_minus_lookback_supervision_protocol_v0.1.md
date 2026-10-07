# B⁻ 层次调用、Lookback 与训练监督协议

## 阶段性设计记录 v0.1

**项目：** AI Modeling  
**讨论整理：** Jack Zhou；ChatGPT（OpenAI）  
**日期：** 2026-10-06  
**状态：** 设计讨论记录；尚无代码、训练结果或性能验证。  
**上游依据：** [第一部分主方案 v1.1](path_control_pretraining_part1_v1.1.md)、[接续说明](HANDOFF.md)及其后的本轮讨论。  

> 本文不替代 v1.1 主方案。它记录 v1.1 之后围绕 B，尤其 B⁻ 的层次调用、Lookback、跨尺度组合性和训练数据监督协议形成的新判断，并给出下一步可验证计划。

---

# 1. 本轮讨论得到的核心判断

第一部分真正最难的算法问题逐渐从“如何做局部可逆 A/D”或“如何让 F 适配粗粒度状态”集中到 **B 的能力边界和训练方法**。

A/D 的职责相对清楚：提供稳定的局部读写坐标变换。F 的训练可能最昂贵，但目标仍接近常规 Transformer：给定当前尺度的状态，完成任务计算。B 则决定：

- 什么内容成为一个父状态；
- 什么细节留在局部 scope；
- 什么时候需要向更细尺度查询；
- 子节点实际执行后，哪些结果需要向上返回；
- 哪些信息应提升为更长寿命的状态；
- 如何在不看未来的情况下学习这些动作。

因此，**B 实际上定义了动态 latent 粒度及不同寿命状态之间的信息交换规则。**

其中 B⁺ 和 B⁻ 的难度并不对称：

- **B⁺ 更像被动归因。** 外部输入已经存在，B⁺ 在当前可见输入上做因果分片、汇合和提升。
- **B⁻ 是主动执行。** 它既要依据上层 F／父状态指导孩子展开，又要观察孩子的真实执行结果，决定是否继续、规约、返回和提升信息；同时还可能承担跨层 Lookback 的查询传递。

这使得 B⁻ 更接近一个 **带 scope 的层次化函数调用协议**，而不只是 decoder 或 unpooling 模块。

---

# 2. 输出执行不是单向展开，而是 Call / Return

早期简化图容易把生成理解为：

```text
F → B⁻ → D → text
```

但真实需求更接近：

```text
上层父状态给出 plan / constraint
        ↓  CALL
B⁻ 创建或选择 child scope
        ↓
下层 B⁻ / D 实际展开
        ↓
child 产生实际结果
        ↑  RETURN
B⁻ 吸收结果，更新当前父 scope
        ↑
必要时继续返回到更高层／F
```

因此一个父节点至少应区分：

- **plan state**：孩子执行前，上层希望完成什么；
- **committed/result state**：孩子执行后，实际发生了什么。

例如 F 可以只决定“生成一个四位编号”，而不必事先决定具体数字。下层实际生成 `3719` 后，可以将“实际编号 = 3719”作为 return payload 向上返回。如果这个结果后来值得长期使用，再提升到更长寿命状态。

这解决了一个重要矛盾：**模型不需要把全部文字重新编码回 F，但下层实际生成的语义结果又不能永久与上层世界状态断开。**

---

# 3. Scope 是 B⁻ 的核心运行语义

B⁻ 的信息流不能只用“父 latent → 子 latent”描述。每一级至少存在三类 scope：

## 3.1 F scope

较长寿命、跨轮可继续使用的状态，包括已提交 latent、任务目标和已经提升的重要结果。

## 3.2 B parent scope

生命周期约等于一个父节点的展开过程，至少包含：

- parent plan；
- 已完成 children 的 return；
- 当前 active child；
- 本级局部缓存；
- 剩余预算；
- stop / close 状态；
- provenance / child reference。

## 3.3 D / 最外层 local scope

更短寿命的实际表达状态，例如局部词序、表面格式、刚产生但尚未提升的数值或标识。

关键约束是：**下层 scope 可以拥有上层尚未长期保存的实际细节，但这些细节不能因此自动进入 F。它们只有在被上层查询、返回并选择提升后才改变寿命。**

---

# 4. 两类 Lookback 暴露了 B 的能力边界

## 4.1 Lookback：用户输入

第一轮用户提供某个细节，例如：

```text
样品 K 的测量值是 17.3，先回复收到。
```

如果 17.3 没进入 F，但仍处于可访问的外围局部缓存，第二轮用户问：

```text
刚才样品 K 的值是多少？
```

当前输入改变了任务需求，B⁺ 可以从尚存的局部输入状态重新提取 17.3，再提升给 F。

这属于 **输入侧 Lookback**：已有外部输入细节被新任务重新归因和提升。

## 4.2 Lookback：模型输出

第一轮模型实际输出：

```text
你的编号是 3719。
```

假设 F 当时只保留“生成一个编号”，具体 `3719` 只存在于下层 B⁻ / D scope。第二轮用户说：

```text
把你刚才那个编号加 1。
```

此时新指令已经进入 F，但 F 自己没有 `3719`。因此需要：

```text
F 发起 QUERY(previous_generated_id)
  ↓
B₁⁻ 检查本级；没有则向下传
  ↓
B₀⁻ / D scope 找到 3719
  ↑
RETURN(3719)
  ↑
逐级返回 F
```

这属于 **输出侧 Lookback**。它说明 B⁻ 必须能够接收上层 query，并在必要时把 query 继续传给更下级 B⁻，再把 return 沿层次向上带回。

这比“B⁻ 只负责展开”更强，也正是 B⁻ 比 B⁺ 复杂的原因之一。

---

# 5. 候选的最小跨尺度信息流协议

本轮讨论提出一个候选的最小 B ABI。其目的不是枚举所有语义动作，而是给跨 scope 信息交换提供稳定原语。

## 5.1 四个候选原语

### EXPAND / CALL_NEW

向下创建新的 child work：

```text
parent plan → child scope
```

### QUERY / CALL_EXISTING

向下访问已有的 child / descendant scope，通常用于 Lookback 或查找尚未提升的细节。

### RETURN

把 child 执行结果向 caller 返回：

```text
child result → parent scope
```

建议至少带：

```text
result + provenance + status
```

其中 status 可以表达 `OK / NOT_FOUND / AMBIGUOUS / EXPIRED` 等，而不必为每种失败再引入新 action。

### COMMIT / PROMOTE

把已经存在于较短寿命 scope 的信息提升为更长寿命状态，例如提升到上一级、某个较粗尺度，或最终进入 F 的会话历史。

COMMIT 不等于 CLOSE。一个父节点可以在尚未结束时把部分结果提升，也可以结束而不提升大量表面细节。

## 5.2 CLOSE 更适合作为生命周期语义

当前候选是把 scope 结束表示为 `RETURN(..., final=true)`，而不是单独作为第五个信息流原语。是否最终采用这一编码仍待形式化。

## 5.3 为什么四个原语“接近完备”

若第一部分的 scope 构成一棵同步父子树，那么任意两个 scope 之间的信息转移都可以沿最近共同祖先分解为：

```text
source → RETURN/COMMIT → ancestor → QUERY/EXPAND → target
```

因此 sibling 之间也不需要专门的 COPY 动作：A 返回父 scope 后，父 scope 再用结果指导 B。

**注意：这只是第一部分树状同步信息流的候选完备性，不包含第二部分未来可能需要的 FORK、ROLLBACK、ABORT 等搜索控制。**

---

# 6. 稳定协议可能允许旧 B 完全冻结

这是本轮讨论得到的一个重要候选方向。

假设训练最外层 B₀⁻ 时，同时训练 F₀ 在需要时产生统一的 action/query：

```text
F₀ → QUERY(q) → B₀⁻ → RETURN(r) → F₀
```

以后新增一级：

```text
F₁ → QUERY(q) → B₁⁻
                   ↓ delegate
                 QUERY(q')
                   ↓
                  B₀⁻
                   ↑
                RETURN(r)
```

如果 B₀⁻ 从第一次训练起就按稳定协议工作，那么它不需要知道未来存在 B₁⁻；B₁⁻ 只是新的 caller。

继续新增 B₂⁻ 时同理。

于是理想的渐进训练可以从：

```text
新增 Bₖ → 重新适配所有 Bₖ₋₁ ... B₀
```

变成：

```text
主要训练 Fₖ + Bₖ；旧 B₀...Bₖ₋₁ 作为冻结子程序调用
```

这对整个“剥洋葱”路线的可扩展训练成本意义很大。

---

# 7. 为什么“固定 action token”本身还不够

如果协议只有固定 opcode：

```text
QUERY
RETURN
EXPAND
COMMIT
```

但 action 后面的 payload 仍是某一代 F 的任意私人 hidden，那么接口仍会漂移：

```text
B₀ 会理解 F₀ 的 q，但未必理解 F₁ 的 q。
```

因此真正需要稳定的是：

> **action semantics + argument representation + return representation + scope semantics**

可以把逻辑调用对象写成：

```text
Call = (action, query/content, target/scope, budget/constraints)
Return = (result, provenance, status)
Commit = (payload, destination/lifetime, provenance)
```

这里最大的未决问题是：**query / result 的表示空间是否需要一个独立、跨模型轮次稳定的 B protocol space。**

如果存在这样的 interface space，那么 F₀、F₁、F₂ 都学习把自己的私人 hidden 映射到同一个调用协议，而旧 B 就更可能冻结。

如果不存在，而父子直接交换任意连续 latent，则早先讨论过的 ingress/egress adapter 或邻近下游联合适配仍可能必要。

因此，“旧 B 可完全冻结”目前是重要候选假设，不是已经证明的结论。

---

# 8. 根本训练问题：Teacher 监督怎样进入计算图

本轮最后形成的更根本判断是：

> **B 的训练首先是训练数据格式与监督消费协议问题，而不是网络结构问题。**

由生成训练数据的 AI / teacher 提供的监督信号，不能作为普通 input content 与原始文本一起送入模型，否则训练时模型能看到、推理时却不存在，形成直接泄漏。

因此需要把训练 episode 至少分成两条逻辑通道。

## 8.1 Observation Tape：推理时可见

只包含部署时真实存在的信息，例如：

- 用户输入；
- 当前已经形成的本级状态；
- 合法的上级 call/query；
- 已完成 child 的 return；
- 当前 scope；
- 已提交 F KV；
- 当前预算和合法运行条件。

模型所有正常 forward 决策只能读取 Observation Tape。

## 8.2 Supervision Tape：训练专用

由 teacher 提供，可以包含推理时不存在的信息，例如：

- 候选边界／组合标签；
- 这个位置应该 EXPAND / QUERY / RETURN / COMMIT；
- query 应下传到哪一级；
- child return 应保留哪些语义结果；
- stop / completion 标签；
- 某信息应保留还是允许过期；
- 候选 merge / split 的未来任务价值；
- 最终任务答案或失败原因。

核心约束：

> **Supervision Tape 不进入部署可见状态，只在对应训练事件上产生 loss、teacher forcing 或候选价值评价。**

---

# 9. 同一份监督可以在不同层级消费

Teacher 不必只给最终答案。一个 episode 可以同时产生多尺度 supervision，而每项监督只能在规定事件上消费。

## 9.1 最早：A / 输入编码阶段

Teacher 可以标出实体、数值、关系等，但标签不作为输入 token，而只用于辅助 head：

```text
X → A(X) → probe / auxiliary head → loss(label)
```

## 9.2 中间：B⁺ 规约阶段

Teacher 可以给候选边界、内容单元或 merge/split 比较。B⁺ 自己仍只根据当时可见状态决策，teacher 只用于 loss 或候选价值。

## 9.3 B⁻ 展开 / Lookback 阶段

Teacher 可以产生事件对齐轨迹，例如：

```text
parent plan
→ EXPAND(number-slot)
→ child 实际得到 3719
→ RETURN(value=3719)
→ later QUERY(previous_generated_id)
→ delegate
→ RETURN(3719)
→ 根据后续价值决定是否 COMMIT
```

Student 不能一次看到整条 teacher 轨迹，而应在每个实际事件到达时消费对应 supervision。

## 9.4 最晚：F 层

Teacher 可以监督 F 判断：

```text
当前缺少某个细节 → 应发起 QUERY
获得 RETURN 后 → 执行新的中央计算
```

这样 F 学“什么时候需要操作”，B 学“操作如何执行和传播”。

---

# 10. 必须区分因果动作标签与 hindsight 价值

这是数据协议中的关键防泄漏规则。

Teacher 可以看到完整未来，但 student 在时刻 t 不能看到未来。如果两个 episode 有相同前缀，但未来问题不同，teacher 不能因为某个具体未来而给相同前缀两个互相冲突、又要求 student 当时直接预测的动作标签。

因此 supervision 至少分为两类。

## 10.1 Causally actionable supervision

在当前信息下原则上可决定，可以直接作为动作或状态标签，例如：

- 当前 child 是否完成；
- 当前 scope 是否有查询答案；
- return 是否满足 caller；
- query 是否需要继续下传；
- 某个已看到的结构边界是否关闭。

## 10.2 Hindsight value supervision

只有看未来才能知道，例如：

- `3719` 以后到底会不会被引用；
- 某个输入细节是否值得长期保留；
- merge / split 对未来多个任务的总影响。

这种 supervision 不能直接伪装成当时唯一正确的 `COMMIT / DROP` 标签，而应成为：

```text
keep_value vs drop_value
candidate future loss
expected future utility
```

Student 最终学习的是：

```text
E[future utility | currently visible state]
```

而不是偷看本次样本的真实未来。

“同前缀、多未来”应成为数据构造和验收的基本方法。

---

# 11. 候选的 Episode / Supervision 数据格式

当前建议把一个训练样本抽象为：

```text
Episode = Observable Events + Hidden Supervision Graph
```

Observable Events 按实际运行时序记录；Hidden Supervision Graph 通过 event id 与这些事件对齐。

一个逻辑 supervision item 可以包含：

```yaml
event: B0.return
kind: action_label
available_at: child_complete
target:
  result: 3719
  provenance: generated_id
  status: OK
```

或者：

```yaml
event: F.lookback
kind: action_label
available_at: turn_2_user_complete
target:
  action: QUERY
  semantic_key: previous_generated_id
```

又或者未来依赖监督：

```yaml
event: B0.retention
kind: hindsight_value
available_at: training_only_after_episode
target:
  keep_value: ...
  drop_value: ...
```

字段名尚未锁定；重要的是必须显式记录：

- 监督针对哪个运行事件；
- 监督属于哪种类型；
- 在训练的哪个时刻允许被消费；
- 是否可以进入 teacher forcing；
- 是否只能做 value / candidate comparison；
- 是否绝不能进入部署可见 state。

---

# 12. 逐级训练计划：先用虚拟 caller，再替换为真实上游 B

这套数据协议为“下游 B 训练时还没有上游 B”提供了脚手架。

## 阶段 0：定义协议，不训练完整模型

先明确：

- scope 生命周期；
- EXPAND / QUERY / RETURN / COMMIT 的严格语义；
- query / return / commit payload；
- provenance 和 addressing；
- supervision event schema；
- causal vs hindsight 标签规则。

## 阶段 1：训练最外层 B₀⁻ + F₀

由 F₀ 和 teacher/oracle caller 混合产生 call：

```text
F₀ / oracle → protocol → B₀⁻ → D → return
```

重点任务：

- 普通展开；
- 本 scope lookback；
- 模型自产细节 lookback；
- NOT_FOUND / AMBIGUOUS；
- return provenance；
- 局部结果提升与不过度 COMMIT。

同时通过 caller diversification 防止 B₀ 只识别 F₀ 的“口音”。

## 阶段 2：冻结 B₀，加入 F₁ + B₁⁻

训练：

```text
F₁ → B₁⁻ → frozen B₀⁻ → D
```

B₁⁻ 学习：

- 本级能回答则直接 RETURN；
- 本级没有则把 QUERY 转成合法下级 call；
- 吸收 B₀ return；
- 决定继续执行、返回 F₁ 或 COMMIT。

这一阶段直接测试“旧 B 可冻结”的核心假设。

## 阶段 3：逐级增加 B₂、B₃……

每新增一级优先只训练：

```text
Fₖ + Bₖ
```

旧 B 作为冻结子程序。如果性能明显退化，再局部开放最近一级下游 B；不能一开始就默认重训全部外围。

## 阶段 4：与 B⁺ / 粗粒度 F 继续预训练结合

确认 B⁻ 的组合性和协议可冻结后，再把它放回完整“剥洋葱”循环：

- 新 B⁺ 学新的粗粒度汇合；
- Fₖ 适配新的父节点分布；
- Bₖ⁻ 学本级展开和调用；
- 已成熟外围 B⁻ 尽量冻结。

---

# 13. 最重要的能力边界实验

下一阶段不应一上来扩大模型，而应先做能够证伪接口假设的小实验。

## 13.1 Input Lookback

输入细节没进 F，但仍在局部读入缓存；新问题到来后必须重新提取，不得无故补问。

## 13.2 Model-output Lookback

模型输出具体随机数／标识只在 D / B₀ scope；下一轮引用时 F 发 QUERY，下层返回真实值。

## 13.3 Nested Lookback

答案位于 B₀，但 F 与它之间隔着 B₂、B₁。query 必须跨多级向下传播，return 再逐级向上。

## 13.4 Sibling dependency

child A 的真实结果返回 parent 后，parent 用它指导 child B；验证不需要 sibling 直连。

## 13.5 Late promotion

某细节第一轮不值得进入 F，第二轮用户引用后价值上升；查询回来后能够 COMMIT，后续长间隔仍可使用。

## 13.6 True absence

所有合法 scope 都没有答案时，RETURN `NOT_FOUND`，最终系统补问，而不是生成性猜测。

## 13.7 Frozen-callee compositionality

训练完 B₀ 后冻结；引入 B₁ / F₁，只训练新层。如果新 caller 能可靠调用旧 B₀，说明稳定 ABI 有效。若必须频繁重训 B₀，则该假设失败。

## 13.8 Same-prefix / multiple-futures

完全相同的前缀配多个不同未来问题，检查 retention / boundary 策略没有利用未来标签泄漏。

## 13.9 Supervision-leak test

去掉全部 Supervision Tape 后运行部署路径，输出质量和路由逻辑必须仍成立；训练专用 annotation 不得成为隐形输入。

---

# 14. 当前尚未解决的关键问题

1. **四个 action 是否最终足够。** 当前仅认为对第一部分树状同步信息流“接近完备”，尚需形式化状态机验证。
2. **query/result 的稳定表示空间。** 是否建立独立 B protocol latent space，是旧 B 能否真正冻结的核心。
3. **scope addressing。** QUERY 如何指定 current / sibling / descendant / recent-output 等目标，而不变成全局搜索。
4. **provenance。** RETURN 和 COMMIT 应保留多少来源和版本信息。
5. **lifetime。** COMMIT 应提升到哪一级、保留多久，如何防止一切都被保险式写入 F。
6. **B⁻ 与 D 的边界。** D 可以自由决定哪些表面细节；哪些具体化必须形成可返回的 semantic result。
7. **B⁺ 与 B⁻ 的协议共享范围。** 两者都处理层次状态，但 B⁺ 是外部输入归因，B⁻ 是主动调用与执行，不能为了形式对称强行共享全部机制。
8. **Teacher 轨迹质量。** 生成数据 AI 能否稳定地产生 event-aligned、无未来泄漏的 supervision graph。
9. **Hindsight value 的估计方法。** 特别是长期 retention 和 merge/split 的未来效用，最终可能需要候选对照、价值模型或后续 RL，但不应过早扩大到完整树搜。
10. **冻结失败时的退路。** 如果 frozen callee 无法适应真实 caller，优先尝试 protocol adapter 或只开放最近一级，而不是重训整个栈。

---

# 15. 下一步工作顺序

本轮讨论改变了下一步优先级。建议不先决定 B 用什么 Transformer / MLP / convolution，而按以下顺序推进：

### 第一步：写正式的 B⁻ 执行状态机

明确：

```text
CALL / QUERY
scope open
local execution
RETURN
optional COMMIT
scope close
```

以及 nested call 的栈行为。

### 第二步：写训练数据协议

定义：

```text
Observable Event schema
Supervision Event schema
available_at
causal label / hindsight value
provenance
scope reference
```

并规定任何 teacher-only 字段如何与正常 forward 隔离。

### 第三步：生成极小可验证 synthetic curriculum

优先覆盖：

- 数字／随机 ID；
- 人名／角色绑定；
- 输入 lookback；
- 输出 lookback；
- 多级 query delegate；
- late promotion；
- 真正缺失后的补问。

这些任务必须程序化校验，不依赖模糊的语言评分。

### 第四步：做 B compositionality 原型

只做两级：

```text
F₁ → B₁ → frozen B₀ → D
```

测试 B₀ 在没有见过真实 B₁ 的情况下，是否能借助早期 oracle-caller 课程被未来 B₁ 正确调用。

### 第五步：决定是否允许旧 B 永久冻结

若两级实验通过，再扩展三级；若失败，定位失败来自：

- action 语义不够；
- payload space 漂移；
- scope/provenance 不足；
- caller 分布变化；
- B₀ 本身过拟合；
- 下层 return 无法组合。

只有定位后才决定增加 adapter 或开放最近一级 B。

### 第六步：回到完整 v1.1 剥洋葱预训练

只有 B 的协议和组合性得到最小验证后，才值得把它和新的 B⁺、粗粒度 F 适配、大规模语言任务结合。否则直接扩大预训练规模，很可能只是昂贵地掩盖接口定义错误。

---

# 16. 当前阶段的结论

本轮讨论形成了三个比 v1.1 更具体的新方向：

**第一，B⁻ 应被视为层次化 call / return / scope 机器，而不是单向展开器。** 它需要支持上层指导、下层实际执行、结果返回，以及跨级 Lookback。

**第二，渐进新增 B 是否可扩展，很大程度取决于能否从一开始定义稳定的跨尺度调用协议。** 如果未来上游只需产生统一 action/query，而旧下游 B 能作为冻结子程序工作，那么每轮的训练范围可能主要限制在新 Fₖ + Bₖ。

**第三，真正的训练脚手架可能不是额外自然语言，而是与运行事件对齐、推理时不可见的 Supervision Tape。** Teacher AI 可以在输入编码、B⁺ 规约、B⁻ 调用／返回和 F action 等不同阶段提供监督，但每项 supervision 必须明确何时允许消费，并严格区分因果动作标签与 hindsight 价值。

因此目前最优先的研究对象不再是“大模型结构应该长什么样”，而是：

> **能否定义一套在训练时可被 teacher 精确监督、在推理时完全自主运行、并能跨尺度递归组合的 B 状态机与数据协议。**

如果这个最小闭环成立，再把它嵌入 v1.1 的多尺度模型系列；如果它失败，应先修协议，而不是靠更大的 F、更多永久缓存或完整树搜来掩盖问题。
