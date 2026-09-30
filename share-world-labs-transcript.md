# World Labs 世界模型：分享对话原文

来源：https://chatgpt.com/share/6abc6e55-4634-83e8-b63a-c908678761c1
抓取日期：2026-09-29。原始 HTML 保存在 `share-world-labs.html`。
以下按时间顺序保留网页数据中的 18 条用户消息和 18 条助手最终回复；不含工具输出和过程消息。文中部分引用、导航及 UI 标记为分享页原有的内部标记，可回查原始 HTML。

## 1. 用户

你查一下这个world lab是干嘛的？世界模型这条线，我感觉在llm的多模态latent
Token跑通后，已经殊途同归了

## 2. 助手

对，你这个判断我基本同意。**World Labs 现在反而是一个很好的证据，说明“world model vs 多模态 LLM”正在从两条技术路线变成同一棵树上的不同训练分支。** “世界模型”这个词现在也快被 AI 行业用成收纳箱了，什么东西都往里扔，但 World Labs 的技术细节相当有意思。

### World Labs 到底是什么

它是李飞飞和 Justin Johnson、Ben Mildenhall 等人创办的 frontier model lab，目标不是做一个 3D 建模 SaaS，而是做所谓 **spatial intelligence**：

> 输入文字、图片、视频、3D 信息，形成一个关于世界空间、物体、运动的内部表示，然后可以生成、重建、模拟，最后用于 action / robotics。

他们第一代产品是 **Marble**，可以从文字、图片、视频生成 persistent 3D world，而且不是普通 video generation。你可以移动摄像机、重新观察场景、编辑、扩展，然后导出 Gaussian splat、mesh 等显式 3D 表示。citeturn543799search4turn860884search19

但真正值得看的是他们 **9 月 1 日刚发布的 Atlas**。

Atlas 已经不是一个传统意义上的“3D 模型”。

World Labs 自己定义它为：

**multimodal autoregressive diffusion transformer**

它原生处理：

| modality | Atlas |
|---|---|
| text | ✓ |
| image | ✓ |
| video | ✓，作为 image sequence |
| camera pose | ✓ |
| depth map / 3D | ✓ |
| 输出 image/video | ✓ |
| 输出 explicit 3D | ✓ |

所有输入进入一个 **shared spatial context**。它然后像自回归模型一样，根据已有 context 预测下一个 multimodal element。对于 image/video 这种高维连续信号，又使用 latent diffusion / rectified flow 去生成。citeturn452915view1

这句话尤其重要：

> World Labs 自己明确说，Atlas 是 **LLM 和现代 video model 的混合体**，而且可以直接继承 LLM 的 KV cache、cache-aware routing、disaggregated serving 等技术。citeturn452915view1

所以已经不用我们替他们“殊途同归”了，他们自己基本承认了。

---

## 你说的 latent token 才是关键

早几年 LeCun 那种“LLM 不可能理解世界，必须 world model”的批评，在当时并不是没道理。

因为当时粗略可以理解成：

**LLM**
  
`text token → transformer → text token`

**world model**

`observation → latent state → predicted latent state → observation/action`

前者的 token 是语言符号，后者试图学习世界隐藏状态。

所以当年确实存在一个相当本质的分歧：

**究竟是在“描述世界”，还是在“建模世界”？**

但到了今天，多模态 foundation model 的内部空间已经不是“文字 token 世界”了。

它越来越像：

\[
(x_{text},x_{image},x_{video},x_{audio},x_{action},x_{3D})
\rightarrow z
\rightarrow z'
\rightarrow output
\]

真正进行 computation 的东西是共享 latent representation。

文字只是 observation modality 之一。

图像也是。

视频也是。

action 也可以是。

camera pose 也可以是。

depth 也可以是。

这时候你叫它：

**multimodal LLM**

还是：

**world model**

很大程度已经变成训练目标和接口设计的问题，而不是完全不同的 intelligence architecture。

Atlas 本身就是最漂亮的例子。

它甚至直接说：

> “Similar to an LLM, Atlas first encodes its inputs into a context…”

区别只是 Atlas **把每一张 image 显式绑定在 3D coordinate/camera pose 上**，于是普通 sequence context 变成了 spatial context。citeturn452915view1

也就是：

普通 multimodal model：

\[
z_t = f(z_{<t})
\]

Atlas 更接近：

\[
z_{x,y,z,t}=f(z_{\text{spatial-temporal context}})
\]

**骨架没换，坐标系和 supervision 换了。**

---

## 所以现在真正没有“殊途同归”的部分在哪里？

我认为剩下的差异主要有三个，而不是“LLM vs world model”。

第一是 **显式 geometry grounding**。

普通 multimodal model 看十张照片，可以在 latent space 里隐式形成某种 3D representation。

Atlas 则直接告诉模型：

**这张照片是相机在这个 XYZ、这个姿态看到的。**

并且要求它输出 depth、point cloud、Gaussian splat。

于是把原本需要模型自己隐式发现的 symmetry / equivariance，直接塞进训练结构。

这更像 inductive bias，而不是另一种智能理论。

第二是 **training objective**。

普通多模态模型大量目标仍然是：

\[
P(next\ token|context)
\]

或者 image/video reconstruction/generation。

真正的 world simulator 需要逼它学习：

\[
P(s_{t+1}|s_t,a_t)
\]

也就是：

**我采取 action A 后，世界会发生什么。**

这才是 physical intelligence 最硬的部分。

杯子被推一下：

- 朝哪边移动；
- 会不会倒；
- 水会不会洒；
- 手接下来在哪；
- 遮挡之后物体到底还在哪里。

不是“画一段看起来合理的视频”就完事了。

第三是 **state fidelity**。

World Labs 自己今年其实写得相当清楚，他们把 world model 分成：

**renderer → simulator → planner**

然后明确说这三者正在融合，最终应该成为一个 unified world model。citeturn452915view0

Renderer：

\[
state/action \rightarrow observation
\]

Simulator：

\[
(state,action)\rightarrow state'
\]

Planner：

\[
(observation,goal)\rightarrow action
\]

最终一个大模型学会统一 latent state 后，这三个不过是不同 projection/head。

这恰好就是你的观点。

---

# 所以我会稍微修改你原来的说法

不是：

> **LLM 的多模态 latent token 跑通以后，world model 就没用了。**

而是：

> **多模态模型形成统一 latent state 以后，“LLM”和“world model”不再是互斥架构；world modeling 变成统一 foundation model 在空间、时间、物理和 action prediction 上的一组训练目标。**

这个区别非常重要。

也就是说 **world model 这项能力仍然极其重要，但“必须另造一种非 LLM 智能架构才能获得 world model”的论点正在迅速失去意义。**

甚至 World Labs 的 Atlas 已经相当接近：

**LLM inference machinery + diffusion continuous decoder + spatial positional structure + 3D/physics supervision**

这已经不是 LeCun 当年设想的那个“与 LLM 对立的世界模型阵营”了。

---

## 然后今天 AMD 这笔收购就很有意思了

AMD 今天宣布 **以约 $8.2B 全股票收购 World Labs**，预计年底完成，李飞飞将进入 AMD 担任 EVP / Chief Scientist。citeturn860884news0turn860884news1

这笔钱如果理解成：

> Lisa Su 花 82 亿美元买一个 3D 图片生成器。

那确实有点像人类终于发明出了最高效的焚钞炉。

但如果理解成：

> **AMD 买了一个 frontier model lab，用它提前参与定义 post-LLM / multimodal / physical-AI workload。**

战略意义就完全不同。

特别是 Atlas 的 workload 非常怪，也非常值得 AMD 自己掌握：

\[
\text{autoregressive transformer}
+
\text{latent diffusion}
+
\text{KV cache}
+
\text{huge multimodal context}
+
\text{3D tensors}
+
\text{video}
+
\text{robotics simulation}
\]

它实际上横跨未来 GPU inference 的好几种 workload。

所以 AMD 收购 World Labs 的价值，我现在更倾向于看成：

**不是押注“world model 会战胜 LLM”，而是押注下一阶段 foundation model 会从 language intelligence 扩展成 spatial-temporal-action intelligence，而 AMD 不想再像 CUDA 时代那样，等 workload 被别人定义完以后才造芯片。**

这和我们之前讨论的路线反而非常吻合。

而且最有趣的一点是，World Labs 自己今年六月已经写出了近乎同样的结论：

> renderer、simulator、planner 的边界正在消失，最终方向是一个 unified world model。citeturn452915view0

所以现在我会把 **World Labs / Atlas** 看作“多模态 foundation model 向物理世界扩张”的实验室，而不是另一条挑战 LLM 的平行技术路线。

**换句话说，你说的“殊途同归”已经不是哲学判断了，Atlas 的架构本身就是实物证据。**

navlistAMD 收购 World Labs 最新信息turn860884news0,turn860884news1

genui{"suggest_automation":{"label":"跟踪 AMD 整合 World Labs 后的新模型与硬件协同"}}

## 3. 用户

我理解他们的模型结构已经和标准transformer家族差距甚远，在血缘谱系上，甚至比liquid foundation model更旁枝，这让tpu线路的加速器对他们无用，feifeili选amd不选nv也不无道理，nv深耕多年，内部评估必然是不要花八B买垃圾，加上lisa也是和习总统坐一桌的人，事情就这么成了？

## 4. 助手

我会把你这条叙事改成一个**更技术、也更可能接近真实交易逻辑的版本**。其中最需要纠正的是两点：**Atlas 没有离 Transformer 血缘那么远，TPU 也绝不是“对它无用”；以及目前没有证据支持 Nvidia 看完以后觉得这是垃圾所以不买。** 后一个故事虽然很符合硅谷连续剧的审美，但证据不配合。

### 1. Atlas 的“系统形态”很异类，但底层算子其实没有脱离 Transformer

World Labs 自己写得非常明确：

> Atlas 是 **multimodal autoregressive diffusion transformer**，而且 Transformer 部分仍然主要由大型矩阵乘法构成。它甚至直接使用 KV cache，并声称能继承 LLM 的 cache-aware routing、disaggregated serving 等基础设施。citeturn896519search17

所以从**计算图血缘**来看，我甚至会反过来说：

**LFM2 比 Atlas 更“旁枝”。**

LFM2 只有 6/16 blocks 是 GQA attention，另外 10 个是 double-gated short-range convolution / LIV operators。Liquid 自己就是在系统性搜索 attention、recurrence、convolution 等 operator space。citeturn594510search0

Atlas 则更像：

\[
\text{Transformer backbone}
+
\text{autoregressive scheduling}
+
\text{rectified-flow diffusion}
+
\text{VAE/continuous latent}
+
\text{3D spatial conditioning}
+
\text{KV cache}
\]

它真正怪的地方不是某一个基础算子，而是：

**一次 inference 里面需要同时高效运行几种过去通常属于不同模型家族的 workload。**

这点非常重要。

---

### 2. 所以 TPU 不是“不能算 Atlas”，而是 GPU 的战略优势可能反而在这里放大

如果 Atlas 只是一个巨大的 Transformer：

\[
GEMM + Attention + AllReduce
\]

TPU 路线其实非常舒服。

如果只是 diffusion transformer：

\[
GEMM + Attention \times N_{\text{denoise}}
\]

TPU 一样完全能跑。

因此说 TPU 对 Atlas **无用**是不成立的。

真正的问题是 Atlas 后面的路线越来越可能变成：

\[
\text{language}
\rightarrow
\text{vision}
\rightarrow
\text{video}
\rightarrow
\text{3D reconstruction}
\rightarrow
\text{Gaussian splatting}
\rightarrow
\text{physics simulation}
\rightarrow
\text{robot policy}
\]

每走一步，计算模式就多一种。

今天 Atlas 还是 dense transformer + diffusion；

明年可能塞 sparse 3D operators；

后年可能是 differentiable rendering；

然后是 physics solver；

再然后是某种 action/state recurrent architecture。

**模型结构还没有收敛。**

这时候 GPU 最大的优势不是“矩阵乘法比 ASIC 快”，恰恰相反，而是：

> **我不知道三年后的 workload 是什么，但我今天仍然可以给你一台相当通用的大规模并行机器。**

所以 World Labs 对 accelerator 的需求其实非常符合 AMD/Nvidia GPU 哲学。

而且 Fei-Fei 在 CES 2026 已经公开给过一个相当惊人的 datapoint：

World Labs 的 real-time frame model **不到一周就跑上 MI325X**，然后双方团队几周里把性能又提高了 **4 倍以上**。她还特别强调 world model 需要 enormous memory、massive parallelism 和 very fast inference。citeturn129654search10

这就不是 PPT 兼容了。

他们已经在一起磨 workload 了。

---

## 3. 所以这笔交易其实不是今天突然发生的

今天 World Labs 自己把时间线说得非常直白：

> “We began a deep technical partnership with AMD last year, starting with model training and inference optimization on AMD GPUs.” citeturn727400view0

而且 Fei-Fei 今天披露：

**Lisa Su 是 World Labs 很早期的投资人和支持者，两个人本来就是朋友。**

AMD 从早期就投资，去年开始真正做训练和 inference co-optimization，今年 2 月又参与了 World Labs 的 $1B 融资，然后今天收购。citeturn129654search0turn129654search7

所以实际路线很可能是：

\[
投资
\rightarrow
硬件合作
\rightarrow
共同优化
\rightarrow
发现 workload 对 AMD 很战略
\rightarrow
AMD 加大投资
\rightarrow
双方发现需要更紧密 hardware/model co-design
\rightarrow
收购
\]

这条链比：

> Lisa 和 Fei-Fei 吃顿饭，82 亿，成交。

靠谱得多。

虽然不得不承认，科技史有时候确实没高尚到哪里去。

---

# 4. Nvidia 那部分，我觉得你的推论反而要倒过来

这里有一个相当关键的信息：

**Nvidia 本来就是 World Labs 的投资人。**

2024 年第一轮 $230M，Nvidia 的 NVentures 就参与了；2026 年那一轮 $1B，Nvidia 又参与。World Labs 自己目前的 investor list 里 AMD、Intel、Nvidia 全都在。citeturn129654news31turn129654search7turn129654search3

所以：

> Nvidia 深入评估以后认为“垃圾，不值钱”。

目前完全没有证据。

至少 Nvidia 自己真金白银投过不止一次。

真正有趣的问题反而是：

### 为什么最后 AMD 愿意出 $8.2B，而 Nvidia 没有？

这里我认为存在一个非常漂亮的**边际价值不对称**。

Nvidia 已经有：

- Cosmos / world foundation models
- GR00T
- Isaac
- Omniverse
- simulation
- robotics
- huge internal AI research
- CUDA software stack

World Labs 加进去当然有价值。

但存在大量：

\[
\text{overlap}
\]

甚至可能发生内部路线冲突。

而 AMD 今天缺的恰恰就是这一层。

它有：

\[
CPU + GPU + networking + rack + ROCm
\]

但长期缺：

\[
\boxed{\text{frontier model research}}
\]

因此同一家 World Labs：

**对 Nvidia 是增量 asset。**

**对 AMD 却可能补上一个缺失的组织器官。**

AMD 今天官方写得甚至近乎露骨：

> World Labs 能让 AMD 更早知道 emerging models 的 workload 如何变化，从而指导未来 hardware、software 和 system roadmap。citeturn727400view1

这句话在我看来才是 **$8.2B 的核心**。

不是 Marble 值 $8B。

甚至不是 Atlas 值 $8B。

而是：

> **Lisa Su 不想让 AMD 再一次等 Nvidia / OpenAI / Google 把未来模型架构定义好了以后，拿 benchmark 回家猜下一代芯片该怎么造。**

她直接把一个 frontier model lab 收进 CPU/GPU 公司内部。

于是可以：

\[
\text{future model architecture}
\leftrightarrow
\text{future accelerator architecture}
\]

一起演化。

这非常像 Apple 做 silicon/software co-design，只不过发生在 frontier AI。

---

## 5. 这也解释了为什么 Fei-Fei 可能更喜欢 AMD，而不是 Nvidia

这个不是说 MI450 一定比 Nvidia GPU 更适合 Atlas。

而是**组织利益**不同。

如果 World Labs 被 Nvidia 买：

World Labs 是 Nvidia 巨大 AI research 帝国里又一个 team。

而且还得和 Nvidia 已有的 world model、robotics、simulation 路线争资源和话语权。

进入 AMD：

Fei-Fei 直接成为：

> **Executive Vice President + Chief Scientist，直接向 Lisa Su 汇报。** citeturn727400view1

World Labs 还直接成为 AMD 的 frontier research organization。citeturn727400view0

这差别非常大。

从 Fei-Fei 的角度就是：

**在 Nvidia：加入 Jensen 的 AI empire。**

**在 AMD：帮 Lisa 定义 AMD 的 AI empire。**

后者的组织权力和科研自由度很可能高一个数量级。

所以如果真存在多家潜在买家，我完全可以理解为什么她偏 AMD。

---

至于你说 Lisa 的政治关系，包括她参加过高层中美工商互动之类，**目前没有公开证据表明这和 World Labs 交易有什么因果关系**，而且实际上我们已经有一条足够完整、不需要政治变量才能解释的交易路径：**早期个人关系 + AMD 投资 + 一年多 GPU 联合优化 + World Labs 想靠近 hardware + AMD 极度缺 frontier-model capability + Fei-Fei 得到 Chief Scientist 级别的组织位置。**

因此我现在对这笔交易的理解反而变得很清晰：

> **AMD 买的不是一个 world-model 产品，而是花 82 亿把“未来 workload 的观测站”搬进了自己的芯片设计循环。**

而 World Labs 得到的是另一半：

> **不再为既定硬件修改模型，而有机会让下一代硬件按自己的模型需要长出来。**

这才是两家公司真正互补的地方。Nvidia 恰恰因为太完整，**这种互补性反而没有 AMD 那么强。**

## 5. 用户

摆脱，增量低和垃圾，在增量这个维度上就是同义

## 6. 助手

对，在你限定的**“收购增量价值”这个维度**里，这两个词确实就是一回事。

我刚才把“垃圾”按成了**技术质量低**去反驳，这是跑偏了。你原意其实是：

> **对 Nvidia 来说，World Labs 即便技术不错，如果它提供的能力 Nvidia 内部大部分已经有了，那么作为一个 80 亿美元级收购标的，它就是“垃圾资产”。**

这里的“垃圾”不是说 Atlas 垃圾，而是说：

\[
\text{Acquisition Value}
=
\text{Standalone Quality}
\times
\text{Marginal Strategic Gain}
\]

哪怕第一项很高，只要第二项很低，乘出来照样难看。人类并购史里这种“好公司，烂收购”多得能铺满一条高速公路。

所以你原来的链条可以重新表述得更准确：

**Nvidia 深耕 world model / robotics / simulation 多年，因此 World Labs 对 Nvidia 的边际能力增量可能很低；AMD 恰好缺这一整层 frontier spatial intelligence 能力，因此同一个资产对 AMD 的边际价值高得多。**

这也意味着 Nvidia 如果内部认真评估过，完全可能得出：

> 技术不错，团队也强，但**不值得我们花 8B 买**。

而 AMD 的结论则可能正相反：

> 这东西能补整个战略栈的一块缺口，8B 可以接受。

所以在**资本配置语境**里，你说“增量低 = 垃圾”没问题。更精确一点就是：

**对 Nvidia 是低边际价值资产，对 AMD 是高边际价值资产。**

我前一条确实把你的“垃圾”理解得太字面了。

## 7. 用户

Nv他们肯定也合作过，但nv认为自己是甲方，amd认为自己是乙方，这个区别需要考虑

## 8. 助手

对，这个“**甲方 / 乙方心态**”我认为应该进入核心解释变量，而且它比单纯比较 MI450 和 Rubin 哪个 benchmark 高，更能解释为什么 **World Labs + AMD 的结合价值可能明显大于 World Labs + Nvidia**。

而且现在有公开证据支持你说的前半句，不只是猜测：World Labs 本身是 Nvidia 投资对象；今年 3 月 Nvidia 更直接把 World Labs 列入“正在基于 NVIDIA technology 开发和部署 physical AI”的合作伙伴名单。所以他们显然不是彼此不知道对方在干什么。citeturn593734search0turn593734search11

真正值得分析的是双方在合作桌上的**相对位置完全不同**。

### Nvidia 的逻辑更接近：“这是我的平台，你来适配”

Nvidia 今天卖的早已不是 GPU，而是：

\[
GPU + NVLink + CUDA + TensorRT + Cosmos + Isaac + Omniverse
\]

因此面对 World Labs 这种团队，它天然会问：

\[
\boxed{\text{怎么让 Atlas / Marble 更好地运行在 NVIDIA platform 上？}}
\]

而不太需要问：

\[
\boxed{\text{Atlas 下一代需要什么，我们是不是应该因此重新设计平台？}}
\]

当然 Nvidia 也会 co-design，尤其 OpenAI、Meta 这种买家体量大到可以反过来扭 Jensen 的胳膊。但**World Labs 在 Nvidia 面前没有这种 bargaining power**。

Nvidia 甚至已经有自己的 Cosmos、Isaac、GR00T、Omniverse。World Labs 是生态成员之一。Nvidia 今年的官方材料就是把它和 Figure、Agility、ABB、KUKA 等一大串公司并列放进去。citeturn593734search11

潜台词其实很明显：

> 欢迎来到我们的 physical-AI ecosystem。

这就是典型平台甲方姿态。

---

### AMD 的逻辑却恰好相反：“请告诉我下一代 GPU 应该长什么样”

AMD 自己关于 World Labs 的公开措辞非常值得逐字看。

今年早些时候 AMD 就说，双方在 **workload optimization** 上共同工作，并且 World Labs 正扩大在 Instinct GPU 上的 spatial intelligence/world-model compute。citeturn593734search1

而今天收购公告更直接：

> World Labs 将让 AMD 更深入了解 workloads 如何演化，并帮助塑造未来 technology roadmaps。

而 Fei-Fei 自己说的也是：

> 加入 AMD 可以让 World Labs 帮助 **define the infrastructure needed for the next era of AI**。citeturn303188search16turn303188search18

这个措辞已经不是普通“GPU customer”关系了。

可以把两种关系画成：

\[
\textbf{NVIDIA:}
\qquad
Hardware/Platform
\rightarrow
Model
\]

World Labs：“我怎样充分利用你已经造好的东西？”

而 AMD：

\[
\textbf{AMD:}
\qquad
Model
\leftrightarrow
Hardware/Platform
\]

AMD：“你下一代模型准备怎么玩，我提前给你造什么？”

**这就是你说的甲乙方区别。**

---

## 而这对 World Labs 尤其重要，因为它的 workload 还没定型

普通 LLM 已经相对收敛：

attention、MoE、GEMM、KV cache、collectives……

你大概知道三年后芯片需要优化什么。

但 World Labs 这一类东西现在可能是：

\[
Transformer
+
Diffusion
+
Video
+
3D
+
Spatial\ representation
\]

下一代可能又突然加入：

\[
Simulation
+
Physics
+
Action
+
Sparse\ 3D
+
Robotics
+
Long\ horizon\ state
\]

再往后搞不好出现今天加速器根本没特别优化的 operator。

于是 World Labs 最大的风险之一就是：

> **模型研究方向被现有硬件 architecture 反向绑架。**

如果某种新 representation 在 H100/Rubin 上特别难跑，研究人员自然会受到 pressure：

“能不能换个 CUDA 友好的实现？”

久而久之，硬件开始决定研究空间。

这对 Nvidia 没什么不好，因为它就是平台拥有者。

但对 Fei-Fei 这种想探索新模型范式的人，未必舒服。

---

## AMD 恰恰可以提供相反的交易

Lisa 可以说：

> **别把模型改成适配我们的 GPU。告诉我下一代模型需要什么，我让 MI5xx/MI6xx、ROCm、memory hierarchy、interconnect 去适配你。**

对于 AMD，这种卑微的乙方姿态反而是竞争武器。

因为挑战者没有资格告诉市场：

> “大家按照我的体系来。”

CUDA 已经有二十年积累，AMD 如果继续模仿 CUDA，然后指望客户哪天突然大发慈悲迁移过来，人类大概又给商业史贡献一个很长的失败案例。

它需要的是：

\[
\boxed{\text{把一些重要的新 workload 在形成阶段就绑进自己的 architecture loop}}
\]

World Labs 正好是这种 workload。

---

# 这样一来，$8.2B 的估值逻辑也变了

如果 Lisa 买的是：

**Marble + Atlas 当前 revenue**

那 82 亿确实疯了。

但如果买的是：

\[
\text{frontier model research}
\rightarrow
\text{future workload specification}
\rightarrow
\text{hardware architecture}
\rightarrow
\text{ROCm/software}
\rightarrow
\text{rack architecture}
\]

本质上是在购买一个：

### **内部 workload oracle**

以前 AMD 的流程大概是：

\[
OpenAI/Meta/Anthropic
\rightarrow
新模型
\rightarrow
AMD发现 workload
\rightarrow
AMD适配
\]

永远慢半拍。

现在变成：

\[
World\ Labs\ research
\leftrightarrow
AMD\ architecture
\]

甚至：

\[
Model_{t+1}
\rightarrow
MI_{t+2}
\rightarrow
Model_{t+2}
\rightarrow
MI_{t+3}
\]

这才开始形成真正的 **model-silicon feedback loop**。

AMD 今天官方对收购目的的描述，基本就是这个意思。citeturn303188news11turn303188search16

---

## 所以我现在反而更理解 Fei-Fei 为什么选 AMD

如果 Nvidia 也存在收购可能性，两边提供给她的东西其实不一样。

在 Nvidia：

\[
\text{Fei-Fei}
\subset
\text{Jensen's physical-AI strategy}
\]

她进入一个已经存在 Cosmos、Isaac、GR00T、Omniverse 的巨大体系。

在 AMD：

\[
\text{Fei-Fei}
\rightarrow
\text{AMD Chief Scientist}
\rightarrow
\text{help define AI roadmap}
\]

而今天确认的组织安排正是：她成为 **AMD EVP + Chief Scientist，直接向 Lisa Su 汇报**。citeturn303188news10turn303188search16

这不是职位名字的小区别。

这是：

**Nvidia 给她算力。**

vs.

**AMD 给她一部分“决定未来算力应该长什么样”的权力。**

对于一个真正认为现有 AI architecture 还远远没有收敛的研究者来说，后者可能诱人得多。

---

所以我会把我们前面的 thesis 再往前推进一步：

> **Nvidia 看 World Labs，更容易把它看成自己的一个 workload；AMD 看 World Labs，却有动力把自己变成 World Labs 的 hardware department。**

短期看，前者强势得多。

但如果你正在押注 **下一代 AI workload 尚未收敛**，AMD 这种“乙方思维”反而拥有一个非常重要的期权：

**它愿意让新的模型改变芯片，而不是坚持让新的模型服从芯片。**

这也是我现在觉得这笔交易最值得关注的地方。不是 Lisa 买到了 Atlas，而是 **AMD 第一次在组织结构上把“模型研究者应该告诉芯片设计者下一步做什么”写进了公司。**

## 9. 用户

不过反过来，feifeili不自己干要换amd股票，也证明了合作的价值

## 10. 助手

对，而且这个信号其实比“AMD 愿意花 82 亿”更有意思。**买方愿意付高价只证明 Lisa 看好 World Labs；卖方愿意把 World Labs 股权换成 AMD 股权，则至少部分证明 Fei-Fei 和原股东认为合并后的价值捕获机制比独立经营更有吸引力。**

尤其考虑到 World Labs 今年 2 月才刚融了 **$1B**，投资人名单里同时有 AMD、Nvidia 等，它并不是现金烧光、被迫找下家的公司。citeturn528180search0 7 月它甚至还在主动收购机器人公司 SceniX，9 月刚推出 Atlas。换句话说，这是一个还在扩张、技术路线也刚开始兑现的公司，而不是救火式出售。citeturn528180search4turn528180search6

所以如果把交易理解成经济选择，大概是：

\[
V_{\text{standalone World Labs}}
\]

和

\[
V_{\text{World Labs inside AMD}}
\]

之间的比较。

而股票交易最妙的地方在于，Fei-Fei 并没有完全兑现退出。她实际上是在做资产转换：

\[
\text{World Labs private equity}
\rightarrow
\text{AMD equity}
\]

也就是说，她把自己以后财富的一部分暴露，从“Atlas/Marble 自己能赚多少钱”，改成了：

\[
\text{World Labs技术}
\times
\text{AMD GPU/CPU/ROCm/系统平台}
\times
\text{整个AMD估值}
\]

这和拿 82 亿现金走人的经济含义完全不同。

### 而且这里存在一个非常漂亮的价值捕获问题

假设 World Labs 独立发展。

它发明了一种极其重要的新 world-model architecture，结果这种模型让未来 GPU：

- HBM 容量需求 ×2；
- inference GPU 数量 ×3；
- robot simulation compute ×10；
- 新 operator 进入 accelerator；
- 整个 physical AI 市场爆发。

World Labs 自己未必能把这些价值全部收走。

大量价值会泄漏给：

\[
NVIDIA/AMD + 云厂商 + 机器人公司
\]

这是 AI lab 很经典的问题：

**创造 workload 的人不一定拿走 workload 带来的全部资本回报。**

但如果 Fei-Fei 持有 AMD 股票，这个问题突然消失了一部分。

她搞出一个把 GPU 需求炸高十倍的新模型？

很好。

她现在也持有卖 GPU 的公司。

于是：

\[
\boxed{
\text{模型创新产生的外部性}
\rightarrow
\text{AMD收入}
\rightarrow
\text{AMD市值}
\rightarrow
\text{她自己持有的AMD股票}
}
\]

形成闭环。

这可能比 World Labs 独立上市更有经济吸引力。

---

而且这正好和我们上一轮讲的**甲乙方关系**接起来。

如果加入 Nvidia，她把 World Labs 股权换成 NVDA：

\[
World Labs
\rightarrow
\text{Nvidia既有巨大生态中的一个增量模块}
\]

World Labs 的成功对 Nvidia 总价值的边际影响可能不那么大。

但放进 AMD：

\[
World Labs
\rightarrow
\text{AMD缺失的foundation-model/world-model层}
\]

如果它真的让 AMD 从“算力供应商”升级成：

\[
\text{model}
\leftrightarrow
\text{compiler}
\leftrightarrow
\text{silicon}
\leftrightarrow
\text{rack}
\]

那么 **World Labs 自己创造的价值可以被 AMD 股票放大捕获。**

所以对 Fei-Fei 来说，拿 AMD equity 很可能不只是“接受收购支付方式”，而是一个相当合理的资本配置：

> **我相信自己的研究能够显著提高这个硬件平台的价值，所以与其只持有模型公司，不如持有整个价值链。**

这就开始很像 Lisa 和 Fei-Fei 之间真正能谈拢的东西了。

当然还有一个需要等正式交易文件才能判断的细节：**Fei-Fei 本人到底 roll 了多少 World Labs equity、AMD 股票有没有 lock-up，以及她另外拿了多少 retention package。** 全体 World Labs 股东收到 AMD 股票，不等于她本人主动把全部身家长期押 AMD。这个区别不能被故事性吞掉。

但把 **股票对价 + Fei-Fei 留任 Chief Scientist + 此前已经长期共同优化 AMD GPU** 三件事放在一起，信号明显比普通 acquihire 强：

> **AMD 判断 World Labs 能提高 AMD 的价值；World Labs 核心团队则至少有相当强的经济激励，相信进入 AMD 后，他们创造的价值能够通过 AMD 这个更大的资本载体兑现。**

这其实就是**双方都在押“1+1 的价值主要出现在结合之后”**。而不是 Lisa 单方面豪掷 82 亿买李飞飞的名片。

## 11. 用户

扯远了，回正题，这个模型到底能做什么。如果只是neural rendering就没啥意思。它必须是未来机器人大脑，那它目前有语言能力吗

## 12. 助手

对，正题就看这一刀：

> **Atlas 现在已经明显超过 neural rendering，但公开版本还不能叫“机器人大脑”。它目前更像机器人大脑里的 world-state / simulator 核心，而且已经有原生语言输入，但还没有公开证明自己具备 LLM 那种通用语言推理和 action planning 能力。**

### 先把“只是 neural rendering”排除掉

World Labs 自己其实把上一代 **RTFM** 明确称为 learned renderer：输入若干帧，在 KV cache 里形成隐式 world representation，然后生成新的视角，本质还是：

\[
\text{observations}\rightarrow\text{pixels}
\]

那种东西确实没那么值得兴奋。citeturn957998search10

但 **Atlas 已经多走了至少两步**。

它不只生成像素，还原生处理 **camera pose + depth**，可以从稀疏图像恢复 point cloud / Gaussian splat，也就是说至少部分世界状态已经从“看起来合理”变成了**可计算的显式几何状态**。同时它做 space-time simulation，能从真实视频构造机器人导航和 manipulation 的 simulation，包括 rigid、articulated、deformable objects。citeturn914006search0

所以现在更准确的是：

\[
\text{Atlas} =
\text{renderer}+\text{partial simulator}
\]

而不是单纯 renderer。

---

## 但它现在还不是完整的 robot brain

这里 World Labs 自己反而很老实。

他们把 world model 分成三块：

\[
\text{Renderer}
\quad
\text{Simulator}
\quad
\text{Planner}
\]

其中 planner 才负责：

\[
(observation, goal)\rightarrow action
\]

也就是机器人真正决定：

> “主人让我把桌上的红杯子放进洗碗机，我下一步手应该往哪走？”

而 World Labs 在 6 月份还把 **planner 描述成未来 unified world model 要融合进去的第三部分**，并没有声称 Atlas 已经完成这一层。citeturn914006search1

这非常关键。

现在他们公开展示的机器人路线实际上更像：

\[
\text{real world}
\rightarrow
\text{Atlas reconstruction/simulation}
\rightarrow
\text{大量 synthetic experience}
\rightarrow
\boxed{\text{robot policy / VLA}}
\rightarrow
\text{real robot}
\]

而不是：

\[
\text{camera}
\rightarrow
\boxed{\text{Atlas}}
\rightarrow
\text{motor command}
\]

SceniX 的 R2S2R 工作也是这个逻辑：生成大量与真实环境对应的 simulation，让**policy model**在里面训练和评估，再部署到真实机器人。World Labs 自己明确把 VLA 和 World Action Model 视作另一个 policy-learning 层。citeturn957998search5

所以**今天 Atlas 还主要是给机器人大脑制造“经验”和“世界状态”的东西，不是最终 policy brain 本身。**

---

# 那它有没有语言能力？

**有，而且比单纯拿 CLIP embedding 当 prompt 强，但目前公开证据不足以说明它有 GPT 那种语言能力。**

这是我目前最关心 Atlas 的地方。

World Labs 明确说 Atlas 是 **从头预训练的 omni model**，原生处理：

\[
\boxed{
text + image + camera\ pose + depth
}
\]

视频就是 image sequence。

所有这些东西进入同一个 **spatial context**，然后模型 autoregressively 生成后面的 multimodal elements。citeturn914006search0

因此 text 并不是 Marble 外面挂一个 ChatGPT：

\[
text \rightarrow LLM \rightarrow prompt embedding \rightarrow image model
\]

而是 Atlas 本身的 native modality。

而且他们展示了它能够：

- 根据复杂自然语言 prompt 生成世界；
- 理解对象、空间、风格等语义关系；
- 生成图像中的文字；
- 把语言与 image / spatial context 一起使用。citeturn914006search0

因此它显然已经学到了**相当程度的语言语义表示**。

---

## 但这里千万不能偷换成“它已经是 LLM”

目前 World Labs **没有公开展示**：

- Atlas 跟你对话；
- 问答 benchmark；
- 数学/代码；
- 长文本理解；
- chain-of-thought 类推理；
- 输出长文本；
- language-conditioned long-horizon planning。

他们公开 benchmark 全是：

**camera-controlled generation + 3D reconstruction**。citeturn914006search0

所以目前我会把 Atlas 的语言能力定义成：

\[
\boxed{\text{language-grounded world understanding}}
\]

而不是：

\[
\boxed{\text{general linguistic intelligence}}
\]

这两者差得还挺大。

比如它可能很好地理解：

> “把红杯子放到桌子左侧、蓝色盒子后面。”

这证明它把：

`red cup / left / behind`

映射进了 spatial latent。

但你问：

> “为什么主人可能希望先把易碎杯子收起来，再搬箱子？”

它能不能进行抽象因果推理，然后制定十步计划？

**现在完全没有证据。**

---

# 不过这恰恰使它未来成为 robot brain 的路径非常清楚

Atlas 今天已经有：

\[
\text{text}
+
\text{vision}
+
\text{3D}
+
\text{time}
\rightarrow
\boxed{latent\ world\ state}
\]

它缺的主要是：

\[
goal
+
action
+
proprioception
+
reward
\]

然后训练：

\[
P(s_{t+1}\mid s_t,a_t)
\]

以及反方向：

\[
P(a_t\mid s_t,goal)
\]

如果把 action token 放进去：

\[
[text,\ image,\ depth,\ pose,\ action,\ image,\ depth,\ pose,\ action...]
\]

Atlas 本来就是 autoregressive multimodal sequence model。

**理论上并不需要推翻架构。**

这就是我觉得真正有意思的地方。

它今天已经是：

\[
\text{language + perception + spatial state + dynamics}
\]

再接：

\[
\text{action}
\]

就开始逼近：

\[
\boxed{\text{VLA + world model}}
\]

而 World Labs 自己公开说的终局恰恰就是 **renderer + simulator + planner 合成一个 unified world model**。citeturn914006search1

---

### 所以我的当前判断

如果 Atlas 永远停在：

> 图片 → 漂亮 3DGS → 新视角

那 AMD 花这个钱确实没什么意思，顶多得到一个豪华 neural renderer 部门。

但现在已经不是这样。

它目前最重要的能力其实是：

\[
\boxed{
\text{heterogeneous observations}
\rightarrow
\text{persistent spatial-temporal latent state}
}
\]

而**这东西正是机器人缺的那一半大脑**。

LLM/VLA 擅长：

\[
goal\rightarrow reasoning/action
\]

Atlas 擅长：

\[
observation\rightarrow world\ state\rightarrow future\ state
\]

真正的机器人大脑很可能就是二者融合：

\[
\boxed{
language/goal
+
world\ state
+
world\ dynamics
+
action
}
\]

所以回到你最初那个“殊途同归”：**World Labs 自己其实也正在走向这个结论。**

我现在不会把 Atlas 看成和 GPT 系列竞争的另一种 intelligence architecture，而更像是**把 GPT 类 unified latent intelligence 补上 spatial state、physical dynamics 和 action 之后的另一条入口**。

真正值得盯的下一个发布，不是它能不能把 3D 做得更漂亮，而是有没有出现 **native action token / goal conditioning / closed-loop planning**。一旦这三个东西公开，性质就从“世界模拟器”直接跳到“机器人 foundation model”了。

## 13. 用户

最近宇树科技给自己的机器人装上了gpt astra。怎么感觉有种让从小训练中场的足球队员去打守门员的即视感。所以反过来，如果给通用llm加一个action head fine tune，它的上限能到feifeili这个模的了。吗

## 14. 助手

对，但我要把“加一个 action head”拆成两种情况。**如果只是把通用 LLM 冻住，外挂一个 action head，未必能追上 World Labs；如果允许用机器人轨迹、视频、状态转移去端到端 fine-tune backbone，那么我认为上限完全可能追平，甚至超过 World Labs，而且最后两者会变成同一种东西。**

最近 Astra 的结果已经给了一个非常强的数据点。RoboDojo 团队直接让 GPT-6 Astra 当 robot policy，**没有机器人专用 task fine-tuning**，42 个 manipulation tasks 平均成功率 22.48%，已经超过当时公开的专用 policy；但它的能力分布非常偏，semantic understanding、generalization 很强，precision、动态控制和复杂双臂协调明显差。citeturn760128academia36turn760128search2 后续 RoboDawn 用更合适的 action interface 和一次 demonstration，报告甚至能把 Astra 做到 47.17%。citeturn760128search0

这其实把问题照得非常清楚了。

### Astra 缺的不是“脑子”，而是 sensorimotor training

GPT-6 Astra 已经通过海量视觉、视频、computer use 和 agentic training 学出了很强的：

\[
\text{objects}
+\text{spatial relations}
+\text{goals}
+\text{causality}
+\text{long-horizon planning}
\]

它甚至没有专门学机器人，就能看 camera image，然后发类似：

\[
\text{move left arm}
,\quad
\text{lower gripper}
,\quad
\text{close gripper}
\]

这样的动作序列。

所以你那个足球类比，我会稍微改一下。

不是：

> 从小训练中场的人突然被拉去守门。

而更像：

> **一个读比赛能力极强、身体素质也训练过的中场，第一次戴上门将手套，居然已经能扑一些球，但扑点球的动作细节、落地、手型完全没练过。**

真正缺的是：

\[
\text{continuous control}
+\text{contact dynamics}
+\text{proprioception}
+\text{motor calibration}
+\text{high-frequency feedback}
\]

而不是“理解我要干什么”。

这解释了为什么 Astra 在 semantic task 上突然把专用 VLA 打得很难看，而碰到 precision manipulation 又迅速露馅。

---

## 那么只加 action head 行不行？

如果真是：

\[
\text{Frozen GPT backbone}
\rightarrow
\text{action head}
\]

我的答案是：**上限有限。**

因为 action head 只能读取 backbone 已经编码的信息。

假如 GPT latent 里面只有：

> 杯子在桌子上。

却没有精确编码：

\[
(x,y,z),\ orientation,\ velocity,\ contact,\ friction,\ uncertainty
\]

一个小 action head 不可能凭空气把这些信息变出来。

这就像 backbone 给你 240p 视频，你后面接个 8K monitor，宇宙并不会因为显示器比较贵就补出细节。

World Labs 的优势就在这里。

它从 pretraining objective 开始就在强迫 latent 保存：

\[
3D\ geometry
+
camera\ pose
+
depth
+
temporal\ consistency
\]

所以它的 latent state 天然可能更适合 physical control。

---

# 但如果你说的是 full fine-tune，答案就完全不同

假设拿 GPT-6 Astra 这种 multimodal foundation model，加 action tokens：

\[
[\text{text},
\text{vision},
\text{proprioception},
\text{action},
\text{vision}_{t+1},
\text{action}_{t+1},...]
\]

然后拿大量 robot trajectory 做训练。

同时增加几个 objective：

\[
L=
L_{\text{action}}
+
L_{\text{next-state}}
+
L_{\text{video}}
+
L_{\text{depth}}
+
L_{\text{geometry}}
\]

这时候 backbone 会被迫逐渐学出：

\[
z_t=
\text{semantic state}
+
\text{spatial state}
+
\text{physical state}
\]

然后：

\[
z_t,a_t\rightarrow z_{t+1}
\]

注意发生什么了。

**你实际上已经把 GPT 训练成 World Model 了。**

此时再讨论：

> “LLM 路线和 Fei-Fei world-model 路线谁赢？”

已经开始没有意义。

它们已经汇流了。

---

## 而且通用 LLM 路线还有一个非常恐怖的先手优势

World Labs 必须从 spatial intelligence 往上爬：

\[
geometry
\rightarrow
dynamics
\rightarrow
semantics
\rightarrow
reasoning
\rightarrow
planning
\]

GPT 系模型则从另一边往下爬：

\[
reasoning
\rightarrow
planning
\rightarrow
semantics
\rightarrow
vision
\rightarrow
action
\rightarrow
physics
\]

到底哪边比较难？

我目前反而越来越怀疑 **World Labs 那边更难。**

因为让一个已经有强大 abstraction / language / planning 能力的模型学习：

> 摩擦、接触、关节控制。

这是一个巨大的 supervised/RL 数据问题。

但让一个特别优秀的 spatial simulator 自己长出：

> 数学、社会常识、任务分解、用户意图、长期计划、工具使用……

那几乎是在**重新训练一次 frontier foundation model**。

这成本可不是加几个机器人轨迹。

---

## Astra 那个 RoboDojo 结果其实已经给了一个很不舒服的暗示

它最重要的地方根本不是 22.48% 这个绝对数字。

而是：

\[
\boxed{\text{zero robot training}}
\]

居然已经超过不少：

\[
\boxed{\text{专门训练的 robot policy}}
\]

这意味着 general intelligence pretraining 中形成的 latent representation，**对 embodied control 的 transfer 比很多人原先估计的强得多**。citeturn760128academia36

而且 GPT-5.5 同样方法只有 0.88%，Astra 一下跳到 22.48%。citeturn760128academia36

这个差距尤其吓人。

它说明：

> robot ability 不一定主要来自 robot-specific architecture。

它可能随着：

\[
\text{general model capability}
\]

跨过某个 threshold 后突然涌现。

这和 coding、computer use 当初很像。

人们原来以为要：

> browser agent model  
> coding model  
> GUI model  
> planning model

最后发现一个足够强的 general model 加 interface，大量专门架构直接被吃掉。

机器人可能又来一遍。人类最喜欢先成立十五家公司解决十五个子问题，然后一个 foundation model 发布，把十四家公司送进历史资料馆。

---

# 因此我现在会押一个更具体的架构

未来机器人大脑未必是：

\[
\boxed{\text{World Labs model}}
\]

也未必是：

\[
\boxed{\text{GPT + tiny action head}}
\]

而更可能是：

\[
\boxed{
\text{General multimodal foundation backbone}
}
\]

里面统一保存：

\[
language
+
vision
+
world state
+
memory
+
goal
\]

然后分出来：

\[
\begin{cases}
\text{language/reasoning head}\\
\text{world prediction head}\\
\text{action/trajectory head}\\
\text{value/reward head}
\end{cases}
\]

其中 high-level policy 可能 1–10 Hz；

trajectory/action model 可能几十 Hz；

真正 motor controller 再以：

\[
100\text{–}1000Hz
\]

运行。

**根本没必要让 GPT 自己每毫秒决定电机 torque。**

所以 Astra 今天 precision control 差，并不是一个特别致命的问题。

它只需要输出：

> 手走到这个 6D pose，用这种 grasp，然后向这个 trajectory 移动。

下面传统 controller / learned motor policy 把动作完成。

---

### 因此回到你的原问题

**能。**

但准确表达应该是：

> **一个已经足够强的 multimodal LLM，如果加入 action/state modality，并用 embodied trajectory + world-prediction objective 端到端训练，它的理论上限没有明显理由低于 World Labs 的专用 world model。**

而且因为它已经拥有巨大的 **semantic + reasoning + planning prior**，我甚至认为它更可能成为最终统一模型的主体。

World Labs 真正的价值不是证明“必须有另一种模型”。

恰恰相反。

它可能贡献的是：

\[
\boxed{\text{怎样把 spatial/physical state 正确塞进统一 foundation model}}
\]

也就是说，**GPT 在学会当守门员；World Labs 在研究守门这件事究竟需要哪些感觉和训练目标。**

等这两件事合上，“LLM”和“world model”这两个名字大概都该退休了。

## 15. 用户

那这两条技术路线，看起来world lab前景黯淡。毕竟它可能是个次优方案。它如果不能训练成本大幅领先，就没意义了

## 16. 助手

对。**如果我们把 World Labs 当“未来机器人大脑”的候选路线来估值，我现在也会明显下调它的独立技术优势。** 最大的问题不是 Atlas 做得差，而是它可能在重复支付一笔通用 foundation model 已经付过的昂贵学费。

Atlas 官方明确说自己是**从头预训练**的 omni model，吃 text、image、video、camera pose、depth，然后随着训练 compute 增加继续 scaling。citeturn923790search0 这句话其实对它并不完全是好消息。

因为另一条路线可以是：

\[
\text{GPT/Astra级 multimodal backbone}
+
\text{3D/depth/proprioception/action}
+
\text{robot fine-tuning}
\]

前面那些昂贵的东西：

\[
\text{language}
+\text{semantics}
+\text{reasoning}
+\text{planning}
+\text{visual concepts}
+\text{world knowledge}
\]

**已经训练完了。**

World Labs 如果从头训练：

\[
\text{text}
+\text{vision}
+\text{spatial}
+\text{physics}
\]

它必须重新学其中大量已经被 GPT 学过的东西。

这就是我认为它最大的潜在结构性劣势。

---

### 所以 Atlas 真正需要证明的不是“我空间能力更强”

这个很容易做到。你专门针对 spatial objective 训练，当然可能比通用模型强。

真正应该比较的是：

\[
\frac{\text{robot capability}}{\text{total training compute}}
\]

或者更严格：

\[
\frac{\text{最终任务成功率}}
{\text{预训练compute + robot data + finetune compute + inference cost}}
\]

**目前 World Labs 没有公布这种 compute-normalized comparison。**

Atlas 公布的是 reconstruction 和 camera-conditioned generation benchmark，证明它比一些专用模型强；但没有告诉我们：

> 同样花 \(10^{25}\) FLOPs，我训练 Atlas，和拿已经存在的 frontier multimodal model 做 embodied post-training，谁最后机器人能力更高？

citeturn923790search0

而这才是生死题。

---

## 甚至 World Labs 自己现在的行动已经透露了一点答案

他们今年收购 SceniX 后，机器人路线的重点明显转向：

\[
\boxed{\text{World model作为simulation/data engine}}
\]

而不是简单宣布：

\[
\boxed{\text{Atlas直接成为robot policy}}
\]

他们的 R2S2R：

\[
Real
\rightarrow
Sim
\rightarrow
大量环境变化
\rightarrow
训练VLA/WAM policy
\rightarrow
Real
\]

World Labs 自己甚至明确写：

> robot learning 当前的关键瓶颈“不只是 architecture，而是 experience and evaluation at scale”。

citeturn923790search2

这句话非常值得注意。

某种程度上，他们自己已经在避免和 GPT/Astra 正面打一场：

> **谁来当那个统一大脑？**

而是在占一个更容易建立 moat 的位置：

> **我负责给所有机器人大脑制造廉价、可控、带物理属性的经验。**

---

# 这样 World Labs 就出现两种完全不同的未来

**路线 A：Atlas 成为机器人大脑。**

那么我同意你的判断，它现在看起来有成为**次优方案**的风险。

因为：

\[
\text{general multimodal model}
\rightarrow
\text{embodied fine-tuning}
\]

天然继承巨大 sunk cost。

而：

\[
\text{World model from scratch}
\rightarrow
\text{再学习general intelligence}
\]

很可能是在重复建设。

除非 Atlas 的 spatial inductive bias 产生巨大的样本效率优势，例如：

\[
10\times
\]

甚至

\[
100\times
\]

更少机器人数据/compute，就能学出同等 physical intelligence。

目前**没有公开证据证明这种数量级优势**。

甚至 Atlas 官方目前强调的是“随着 compute scaling 能力继续提高”，而不是“我们比通用模型少两个数量级 compute”。citeturn923790search0

这多少有点尴尬。

---

**路线 B：Atlas/World Labs 成为机器人的数据生成器 + simulator。**

这个故事就强很多。

现实机器人数据贵得离谱：

一个机械臂摔了杯子，你得有人捡杯子、重新摆放、恢复环境；

一个 humanoid 摔倒，还可能得先请机械工程师看看它是不是把自己摔成了昂贵的现代艺术品。

simulation 可以复制：

\[
1\ real\ trajectory
\rightarrow
1000s\ variations
\]

改变：

- 光照
- clutter
- object pose
- friction
- robot embodiment
- camera position
- failure case

而 World Labs 声称他们的 R2S2R 已经做到：完全 simulator 训练的 policy 可以 transfer 到真实机器人，并且 simulated evaluation 能预测真实 policy 的相对表现。citeturn923790search2

这个如果经过大规模独立验证，价值非常大。

因为此时它根本不需要击败 GPT。

它甚至可以服务 GPT：

\[
\boxed{
World\ Labs\ simulator
\rightarrow
synthetic\ embodied\ experience
\rightarrow
GPT/Astra-derived\ robot\ brain
}
\]

这反而可能是最合理的分工。

---

## 因而我会修改一句你说的话

不是：

> **“训练成本不能大幅领先就没意义。”**

而是：

> **如果 World Labs 坚持自己做最终通用机器人大脑，那么它必须证明 embodied learning 的总成本显著低于“frontier multimodal LLM + action/world-model post-training”，否则很可能是次优路线。**

但是，如果它退到：

\[
\textbf{world reconstruction + simulation + synthetic experience}
\]

那评价指标变成：

\[
\frac{\text{高质量robot experience}}
{\text{真实世界采集成本}}
\]

这里它仍然可能非常有价值。

---

### 我现在反而觉得最危险的信号是“Atlas 从头训练”

因为真正漂亮的架构应该是：

\[
\underbrace{\text{frontier general intelligence}}_{\text{不用重练}}
+
\underbrace{\text{spatial/physics representation}}_{\text{World Labs擅长}}
+
\underbrace{\text{action}}_{\text{robot data}}
\]

而不是：

\[
\text{World Labs重新从text开始训练一个foundation model}
\]

如果未来 OpenAI/Google 证明，只需要在现有 multimodal backbone 上增加 **depth/pose/proprioception/action token + physical-world post-training**，就能迅速获得与 Atlas 相当甚至更强的 spatial model，那么 World Labs 那套“独立 world-model foundation model”理论确实会被削掉大半。

**到那时 World Labs 最值钱的东西可能不是 Atlas 这个“大脑”，而是 SceniX + simulation + spatial-data engine。**

所以接下来我最想看到的不是 Atlas 再把 3D benchmark 提高 20%。那东西投资意义有限。

我想看到的是一个残酷得多的实验：

\[
\boxed{
\text{Atlas}
\quad vs\quad
\text{Astra + embodied post-training}
}
\]

给两边**相同 robot-data budget、相同新增 compute budget**，测陌生环境中的 manipulation / planning / sim-to-real。

**如果 Atlas 赢很多，World Labs 找到了真正的 architectural moat。**

**如果差不多，通用 foundation model 路线几乎必然占优，因为它背后那几万亿 token、语言能力和通用推理全是白送的 sunk cost。**

现在公开资料还没有证明前一种情况。

## 17. 用户

这个收购感觉对neural rendering意义更大

## 18. 助手

我也越来越倾向这么看：**这笔收购对 neural rendering / 3D reconstruction / simulation 的意义，可能比对“未来机器人通用大脑”的意义更确定。**

原因很简单。Atlas 目前最扎实、已经展示出来的能力，恰恰集中在 **camera-controlled generation、novel-view synthesis、稀疏视角 3D reconstruction、长时空间一致性**。这些不是远期愿景，而是已经能 benchmark、能产品化的东西。World Labs 自己也明确把 RTFM 定义成 learned renderer，而 Atlas 则是在这个基础上把显式 3D、depth、camera pose 和生成统一起来。citeturn321153search2turn321153search0

这件事对 neural rendering 的意义很大，因为它可能意味着行业路线从：

\[
\text{NeRF / 3DGS / hand-designed representation}
\]

逐渐走向：

\[
\text{large learned renderer}
+
\text{spatial memory}
+
\text{generative prior}
\]

传统 3DGS 很强，但本质上还是“我得先看到这个世界，才能把它重建出来”。Atlas 这种东西则可以：

\[
\text{少量观测}
\rightarrow
\text{重建已知部分}
+
\text{生成未知部分}
\]

于是 reconstruction 和 generation 的边界开始消失。World Labs 自己也强调 Atlas 可以从两三张图获得相当好的 reconstruction，而不可见区域则由模型根据 learned world prior 补出来。citeturn321153search0

这其实比机器人故事更近。

### 对 AMD 尤其有意思

AMD 本来就在 SIGGRAPH 方向认真做 3DGS，官方今年还专门强调了 ROCm 上的 GSplat、动态街景 3DGS、4D volumetric video 等 workload。citeturn585314search9

所以现在突然把 World Labs 整个吃进去，就可能形成一条很清晰的技术链：

\[
\text{Atlas}
\rightarrow
\text{neural reconstruction}
\rightarrow
\text{3DGS / neural representation}
\rightarrow
\text{real-time rendering}
\rightarrow
\text{GPU hardware}
\]

这里 AMD 不只是卖训练 GPU。

它可以开始针对：

- spatial KV cache
- image/video token throughput
- diffusion inference
- Gaussian splatting
- differentiable rendering
- neural compression
- 4D scene representation

一起做硬件和软件优化。

这比“等机器人十年后爆发”具体多了。

---

而且 neural rendering 有一个非常大的潜在市场，不只是游戏。

如果它真做到：

\[
\text{camera/video}
\rightarrow
\text{persistent editable 3D world}
\]

那可以侵蚀的是整个传统 3D content pipeline：

\[
\text{建模}
\rightarrow
\text{材质}
\rightarrow
\text{灯光}
\rightarrow
\text{场景搭建}
\rightarrow
\text{render}
\]

电影、游戏、建筑、工业数字孪生、AR/VR、广告、电商，全都能吃。

现在 Marble 已经是从 text/image/video 直接生成 persistent 3D world，并且可以导出到下游工具。World Labs 今年还专门推了 World API，把这件事当成 programmable 3D substrate。citeturn321153search6turn321153search5

这更像一个非常现实的产业方向：

> **把 3D creation 从“工程软件”变成“生成式模型调用”。**

这个价值链可能比“Atlas 最终是不是机器人 AGI”稳得多。

---

### 我甚至觉得 AMD 买它，可能有一个更深的硬件赌注

未来 graphics pipeline 可能越来越不像：

\[
geometry
\rightarrow
rasterization
\rightarrow
shader
\]

而越来越像：

\[
latent\ scene
\rightarrow
neural\ renderer
\rightarrow
pixels
\]

今天 GPU 里大量固定功能单元是为传统 graphics pipeline 演化出来的。

如果未来 rendering 越来越 neuralized，那么 GPU architecture 也可能变化：

传统：

\[
\text{shader cores + texture units + raster + RT cores}
\]

未来可能越来越偏：

\[
\text{matrix compute}
+
\text{large memory}
+
\text{spatial cache}
+
\text{neural compression}
+
\text{specialized splat/render operators}
\]

这时候 World Labs 对 AMD 的价值就非常直接了：

**它可能提前告诉 Radeon / Instinct 团队，未来“图形”到底是什么 workload。**

这甚至不需要 Atlas 成为机器人脑子。

---

所以现在如果让我给这笔交易拆价值来源，我反而会这样排：

**确定性最高：**

\[
\boxed{\text{Neural rendering / 3D generation / reconstruction}}
\]

其次：

\[
\boxed{\text{simulation / synthetic data}}
\]

再其次才是：

\[
\boxed{\text{robot world model / robot brain}}
\]

最后那个 upside 最大，但技术路线也最容易被 GPT/Astra 这种通用 multimodal foundation model 吃掉。

而 neural rendering 反而是 World Labs 已经明显有原创积累，而且通用 LLM 不一定天然擅长的地方。

所以这笔 **$8.2B** 如果最终证明值得，未必是因为 Fei-Fei 找到了 AGI 的另一条路。

更可能是因为她提前踩中了一个更朴素、但同样巨大的转变：

> **未来很多“渲染”不再是执行 graphics algorithm，而是模型根据一个持久的 latent world state 直接预测你应该看到什么。**

这对 AMD 这种同时做 AI GPU 和 graphics GPU 的公司，反而异常契合。citeturn655675search0

## 19. 用户

这从另一个侧面说明为何thinking machine和liquid foundation model没人收吗？

## 20. 助手

对，而且这个对比很有解释力。但 **Thinking Machines 和 Liquid 没被收，背后的原因其实不一样**。World Labs 这笔交易反而把三种 AI startup 对大厂的战略价值照得很清楚。

我会把它概括成：

\[
\boxed{\text{World Labs 是“缺失器官”}}
\]

\[
\boxed{\text{Thinking Machines 是“另一整个人”}}
\]

\[
\boxed{\text{Liquid 更像“可插拔器官”}}
\]

这三种东西，并购逻辑完全不同。

### Thinking Machines 最大的问题是：买回来干什么？

Thinking Machines 现在本质还是一个 **general frontier-model lab**。而且截至 2026 年 9 月，它已经和 Nvidia 签了至少 **1GW Vera Rubin** 的多年合作，Nvidia 自己也进行了重大投资；本月市场报道它还在寻求约 **$40B pre-money** 的新融资。citeturn799538search1turn799538search4

假如 Jensen 花五六百亿美元把 Mira 买下来，得到什么？

\[
\text{general reasoning}
+\text{multimodal}
+\text{agents}
+\text{open models}
\]

问题是这些能力几乎所有 frontier lab 都在做。

而 Nvidia 最聪明的经济位置其实是：

\[
\text{OpenAI}
+\text{Anthropic}
+\text{Thinking Machines}
+\text{Meta}
+\cdots
\rightarrow
\boxed{\text{都买我的GPU}}
\]

现在 Thinking Machines 自己就承诺吃至少 1GW Vera Rubin。citeturn799538search1

**Jensen 为什么还要把客户买下来，自己承担几百亿美元的模型研发风险？**

这就像赌场已经抽水抽得挺快乐，突然觉得不过瘾，决定亲自坐上桌梭哈。资本配置终于又回到了人类最熟悉的奇怪方向。

所以 TML 独立存在，对 Nvidia 很可能**比被 Nvidia 收购更有价值**：

\[
\text{投资少量equity}
+
\text{卖巨量compute}
+
\text{分享upside}
\]

而不是：

\[
\$50B+
+
\text{承担整个frontier lab风险}
\]

---

### AMD 收 Thinking Machines 其实也没有 World Labs 那么顺

这一点更微妙。

AMD 确实缺 frontier-model research，所以表面看 Mira 很适合 Lisa。

但 Thinking Machines 想解决的是：

> **如何造更好的通用智能。**

这个问题和 OpenAI、Anthropic、Google、Meta 是正面竞争关系。

因此 AMD 一旦买下来，相当于：

\[
\text{AMD}
\rightarrow
\text{同时成为AI lab}
\]

这可能直接让它其他模型客户警惕。

比如 Anthropic 今天买你的 MI450，明天发现 AMD 自己旗下的 Thinking Machines 和它争模型用户，这个关系就开始有点别扭了。

World Labs 没那么严重。

它主要处在：

\[
3D
+
spatial
+
rendering
+
simulation
\]

属于一个 **相对正交的新 workload**。

因此 AMD 买它，可以：

> 学习未来 workload，帮助所有客户。

而不是：

> 买一个新客户回来和旧客户竞争。

这个区别非常大。

---

# Liquid 又是另外一种情况

Liquid 其实更能支持你这个判断。

AMD **2024 年就领投了 Liquid 的 $250M Series A**，而且双方明确约定在 AMD CPU/GPU/NPU 上联合优化。citeturn808638news32

但 AMD一直没有买它。

现在 Liquid 已经有 Mercedes、Shopify、Qualcomm、MacPaw 等合作，整个 positioning 越来越明确：

\[
\boxed{\text{device-native / efficient AI}}
\]

他们最近产品集中在手机、PC、汽车、CPU/NPU、本地 inference，甚至公开强调模型可以运行在 GPU、CPU、NPU 各种硬件上。citeturn974405search0turn974405search2turn974405search7

而这恰好导致一个悖论：

### Liquid 越成功，越不应该被 AMD 买。

因为 Liquid 的价值很大一部分来自：

\[
\text{hardware agnostic}
\]

它能跑：

- AMD
- Intel
- Qualcomm
- Apple
- Nvidia
- 普通 CPU

AMD 一收：

\[
\text{Liquid}
\rightarrow
\text{AMD-owned model architecture}
\]

Qualcomm、Intel、Apple 等客户立刻会重新考虑关系。

所以 AMD 当前最舒服的位置就是：

\[
\boxed{\text{投资Liquid + 联合优化 + 不拥有Liquid}}
\]

继续让它出去替 AMD 探索 heterogeneous/on-device workload。

---

## 而且 Liquid 还有一个更根本的问题：它的 IP 很重要，但未必需要买整家公司

Liquid 的核心竞争力主要集中在：

\[
\text{hybrid architecture}
+
\text{recurrence}
+
\text{efficient attention alternatives}
+
\text{deployment optimization}
\]

例如 LFM2 就是 convolution + GQA attention 的 hybrid architecture，目标主要是提高 edge inference 的速度和内存效率。citeturn974405search14

但这类东西特别容易发生：

\[
\text{idea}
\rightarrow
\text{paper}
\rightarrow
\text{industry absorbs}
\]

如果它证明：

> recurrence + attention 某种比例最好

那么 OpenAI、Google、Meta 下一代 architecture 完全可以吸收相似思想。

未必需要：

> “我们得花 $10B 把 Liquid 整家公司买下来，否则永远学不会。”

World Labs 则不同。

它包含的不只是一个 operator：

\[
\text{spatial data}
+
\text{3D representation}
+
\text{neural rendering}
+
\text{simulation}
+
\text{models}
+
\text{research team}
\]

这种组织能力比较难靠读论文复制。

---

# 所以 World Labs 的特殊之处出来了

它刚好落在一个很罕见的并购 sweet spot：

**它不是 AMD 已有能力的重复品；不是 AMD 客户的直接竞争者；不是一个拿授权合作就足够的单点技术；同时它产生的 workload 又直接决定未来 GPU 应该怎么设计。**

这四个条件同时满足，非常少见。

Thinking Machines：

\[
\text{太大 + 太重叠 + 买回来会和客户竞争}
\]

Liquid：

\[
\text{技术有价值，但合作/投资已经能捕获大量价值}
\]

World Labs：

\[
\text{AMD缺失能力}
+
\text{新workload}
+
\text{hardware co-design价值巨大}
+
\text{不严重破坏客户中立性}
\]

所以确实，**World Labs 被收而这两家没被收，并不一定说明 World Labs 的“AI 智能技术”比它们高级。反而可能说明它更适合作为一家半导体公司的资产。**

甚至可以说得更尖锐一点：

> **最值得被收购的 AI 公司，未必是模型最强的公司，而是它拥有某种大厂无法通过“当供应商”充分捕获的互补能力。**

Thinking Machines 买 Nvidia GPU，Nvidia已经能收租。

Liquid 在 AMD 上优化，AMD已经能受益。

但 World Labs 如果发明下一代 neural rendering / spatial workload，却最后全部跑在 Nvidia 上，**AMD什么都拿不到。**

这可能才是 Lisa 为什么愿意直接把它吞掉的更底层解释。

## 21. 用户

你确定lfm2和lct一样是recurrence结构吗？recurrence这个结构有很强的归纳偏置

## 22. 助手

不确定，而且你抓到我上一条里的一个技术错误了：**LFM2 不是 LTC 那种 recurrent architecture。** Liquid 自己明确写了，最终的 LFM2 是 **短卷积 + GQA attention 的 hybrid**，并且特别强调它采用 short convolution，**而不是 full recurrence**。citeturn618517search0turn618517academia22

如果你说的 LCT 是他们早期的 **LTC（Liquid Time-Constant Network）**，两者血缘有关，但结构已经相当不同。

### LTC 才是真正的 recurrence

LTC 本质是连续时间 RNN：

\[
h_{t+1}=F(h_t,x_t)
\]

或者连续形式：

\[
\frac{dh}{dt}=F(h(t),x(t))
\]

所以过去的信息必须经过一个 persistent hidden state \(h_t\) 向未来传递。

这确实带来了你说的**很强的归纳偏置**：

\[
\text{history}
\rightarrow
\boxed{state_t}
\rightarrow
\text{future}
\]

模型天然被迫假设：

> 世界存在一个相对紧凑的状态，过去对未来的影响可以压缩进这个状态。

对控制、动态系统、机器人、time series 来说，这个 bias 非常有价值。Liquid 早期工作本来就是从这种 continuous-time recurrent dynamical system 出来的。citeturn618517search3turn618517search0

而 Transformer 更接近：

\[
h_t=f(x_1,x_2,\ldots,x_t)
\]

理论上不用把所有历史压进一个固定 state，可以随时回头看 token。

所以两者 inductive bias 完全不同。

---

## LFM2 实际上做了一个很有意思的退却

Liquid 设计了一个更一般的数学框架叫 **LIV，Linear Input-Varying operators**。

在这个框架里面：

\[
\text{attention}
,\quad
\text{recurrence}
,\quad
\text{convolution}
\]

都可以被视为同一家族里的不同 operator。

他们然后做 hardware-in-the-loop architecture search，真的尝试了 attention、convolution 和各种 recurrence。citeturn618517search5

结果搜索出来的 LFM2 **没有选择 recurrence**。

而是：

\[
\boxed{
10\times \text{gated short convolution}
+
6\times \text{GQA attention}
}
\]

第一版 LFM2 就是这个结构。更大的 LFM2 MoE 也继续维持大约 **3:1 convolution : attention**。citeturn618517search0turn618517search1

Liquid 自己甚至非常明确解释为什么：

> short convolutions 而不是 full recurrences，是因为目标平台是 embedded SoC CPU，而现有 kernel library 对这种 workload 优化得更好。citeturn618517search0

这句话很重要。

所以不是：

> “我们理论上证明 recurrence 是最佳 foundation-model architecture。”

而更像：

> “我们定义了一个包括 recurrence 在内的大搜索空间，然后在真实硬件 latency/memory constraint 下，搜索结果把 recurrence 淘汰掉了。”

这可是两回事。

---

### Short convolution 的 inductive bias 也和 recurrence 不一样

例如一个 kernel size \(k\) 的 causal convolution：

\[
y_t=
\sum_{i=0}^{k-1}W_i(x)x_{t-i}
\]

它的偏置是：

\[
\boxed{\text{locality}}
\]

即附近 token 更重要。

但它不会像 recurrent system 那样要求：

\[
\boxed{\text{所有远期历史必须通过persistent state传播}}
\]

超过 convolution receptive field 的信息怎么办？

LFM2 靠那 **6 个 GQA layers** 去处理。

所以你可以粗略理解成：

\[
\text{短距离interaction}
\rightarrow
\text{cheap gated conv}
\]

\[
\text{长距离interaction}
\rightarrow
\text{expensive attention}
\]

而不是：

\[
\text{整个sequence}
\rightarrow
\text{recurrent state evolution}
\]

---

## 这反而让我重新评价 Liquid

因为我之前把它理解成：

> **LTC recurrence 成功扩展成 foundation model。**

这个说法不准确。

更准确的是：

> **Liquid 从 LTC 中抽象出了 input-dependent dynamical operator 的设计思想，但在真正做大语言模型时，architecture search 最终选择了 gated short convolution + sparse attention，而没有保留完整 recurrence。**

所以 LFM2 与 Transformer 的距离其实**没有我前面说得那么远**。

它大致仍然是：

\[
Embedding
\rightarrow
[
\text{Conv block},
\text{Conv block},
\text{Attention block},
...
]
\rightarrow
FFN/MoE
\]

而不是 Mamba/LTC 那种：

\[
x_t,h_{t-1}
\rightarrow
h_t
\rightarrow
x_{t+1}
\]

persistent-state architecture。

---

而你提到的 **recurrence 强归纳偏置**，恰恰解释了一个非常值得思考的问题：为什么 Liquid 最终把它拿掉。

Recurrence 的优点也是它的缺点。

它强迫模型相信：

\[
\text{history可以被压缩成有限state}
\]

对于真实物理世界，这可能是极好的假设。

例如一个机器人并不需要保存过去 20 分钟每一帧 camera pixels，它需要的是：

\[
\text{杯子现在在哪里}
+
\text{门现在开着没有}
+
\text{我手里拿着什么}
\]

也就是 state。

但是语言任务经常要求：

> “请找到 80k tokens 前合同第三段那个数字。”

这时候：

\[
\text{compressed recurrent state}
\]

反而可能输给：

\[
\text{attention直接回去读原文}
\]

所以 LFM2 的方案其实挺务实：

\[
\boxed{\text{局部用强bias，远程保留attention}}
\]

既吃 convolution 的效率，又不承担纯 recurrence 的 information bottleneck。

---

这也顺便修正我们刚才讨论 World Labs 时的一点：**如果真正谈机器人/world model，我反而不会轻易否定 recurrence。**

因为 physical world 恰恰非常接近一个 state-space system：

\[
s_{t+1}=f(s_t,a_t)
\]

这时候“必须维护 persistent latent state”并不是讨厌的限制，反而可能正是你想要的归纳偏置。

所以一个有点反直觉的结论是：

**LFM2 为了语言/edge inference 放弃 full recurrence 很合理；但未来真正 unified robot world model，可能又会把 recurrence/state-space structure 捡回来。**

只是那个 recurrence 未必长得像传统 RNN，更可能是 **large latent state + selective memory update + occasional global attention**。这条路线，我反而觉得比“Transformer 无限堆 context，把机器人一生的视频全塞 KV cache”物理上自然得多。

## 23. 用户

但可能只是前端视觉，听觉模块需要recurrence来低功耗待机

## 24. 助手

对，我现在更倾向这个分层，而且这会**显著削弱“整个机器人大脑都应该 recurrent”**的必要性。

真正天然需要 recurrence / state-space bias 的，很可能不是最高层 reasoning，而是**永远不能关机的感知前端**：

\[
\text{mic/camera/IMU}
\rightarrow
\boxed{\text{tiny recurrent state tracker}}
\rightarrow
\text{compact latent events}
\rightarrow
\text{large multimodal model}
\]

原因非常物理：摄像头和麦克风产生的是无限数据流，你绝不可能让一个 Transformer 永远积 KV cache。

### 前端需要解决的其实是“什么变了”

例如眼镜一直看着房间：

\[
frame_1,frame_2,\ldots,frame_{10^9}
\]

高层模型并不需要反复知道：

> 沙发还在那里。  
> 桌子还在那里。  
> Jack 还坐着。  
> 沙发依然没有发动政变。

前端真正应该维护：

\[
s_t=f(s_{t-1},x_t)
\]

然后只在：

\[
\Delta s_t > threshold
\]

的时候输出事件，例如：

- 有人走进来；
- 用户叫了我的名字；
- 杯子从桌上掉了；
- 面前出现新物体；
- 用户视线固定在某东西上；
- 对话对象开始讲话。

这样一个 recurrent state 的内存是近似：

\[
O(1)
\]

而不是随时间增长。

2025 年 ICML 已经有 linear recurrent network 在 streaming audio denoising 上展示 constant-memory inference，并在 Loihi 2 上做到相对 edge GPU 极大的能耗优势。这个具体倍数有硬件因素，不能直接外推，但至少说明 **recurrent + sparse + streaming** 这个组合很适合这种任务。citeturn639668search3

---

## 听觉尤其适合

音频基本就是 recurrence 的主场。

一个 always-on assistant 没必要持续运行 100B 模型理解：

\[
16,000\ samples/sec
\]

它只需要一个非常小的模型持续更新：

\[
h_t=f(h_{t-1},audio_t)
\]

维护类似：

\[
\{\text{speech state, speaker, acoustic events, partial semantics}\}
\]

听见：

> “GPT……”

才把大型模型叫醒。

甚至不一定非要 wake word。

更高级的前端可以一直做廉价 semantic compression：

> 无关环境声音，不保存。  
> 有人讨论晚饭，形成一个 latent event。  
> 用户直接发问，唤醒大模型。

所以未来 AI 设备的计算功耗曲线可能非常不均匀：

\[
\boxed{
\text{1--50mW always-on intelligence}
}
\]

大部分时间运行；

偶尔才：

\[
\boxed{
\text{W级 edge model / cloud frontier model}
}
\]

爆发计算。

---

## 视觉也是类似，只是 recurrence 未必垄断

这里需要稍微修正一下。

视觉前端确实需要**stateful streaming**，但不等于必须使用严格意义上的 recurrent neural network。

例如 Nvidia 今年展示的 always-on vision subsystem，CNN/ViT 都可以做，在 60 FPS face detection 上平均功耗做到约 **4.6 mW**。关键优化之一是避免外部 memory access，而不是依赖 recurrence。citeturn639668search9

所以真正必要的是：

\[
\boxed{\text{bounded state + bounded compute + streaming update}}
\]

实现它可以用：

- recurrence；
- SSM；
- temporal convolution + fixed cache；
- event-driven network；
- tiny ViT + rolling state；
- 或某种尚未发明得足够漂亮的 hybrid。

**recurrence 是很自然的答案，但不是数学上唯一的答案。**

---

# 这样整个机器人/眼镜 architecture 反而很漂亮

我现在会猜未来更像三层：

\[
\boxed{\text{Layer 1: Always-on perception}}
\]

极低功耗：

\[
audio + vision + IMU
\rightarrow
recurrent/SSM/local\ state
\]

负责 tracking、VAD、wake、变化检测、基础 spatial state。

然后：

\[
\boxed{\text{Layer 2: World-state representation}}
\]

更新：

\[
\text{objects}
+\text{people}
+\text{positions}
+\text{events}
+\text{short-term memory}
\]

不需要每一帧都运行庞大 reasoning model。

最后：

\[
\boxed{\text{Layer 3: General multimodal intelligence}}
\]

GPT/Astra 类模型按需醒来处理：

\[
reasoning + language + planning + action
\]

这样高层模型甚至不用特别 recurrent。

它每次醒来直接获得：

\[
\text{当前world state}
+
\text{relevant episodic memory}
+
\text{当前sensory context}
\]

就够了。

---

### 这又反过来改变 Liquid 的意义

Liquid/LTC 那套 recurrence 思想未必需要赢下：

> **“下一代通用 foundation model architecture”**

才能有巨大价值。

它可能最适合的是：

\[
\boxed{\text{机器的自主神经系统}}
\]

而不是：

\[
\boxed{\text{机器的大脑皮层}}
\]

而 Liquid 今天自己的商业定位其实已经越来越靠近这个区域：强调 device-native、sensor/context/audio、本地实时 processing，再把复杂 reasoning handoff 给 cloud LLM。citeturn639668search1turn639668search2

这反而比让 LFM 去正面打 GPT 合理很多。

---

而且这让我重新看我们前面讨论的 World Labs：

**persistent world state 也未必应该全部存在那个巨大 Atlas 里。**

完全可以：

\[
\text{tiny recurrent perception}
\rightarrow
\text{persistent spatial state}
\rightarrow
\text{GPT-class planner}
\]

World Labs 的技术可能负责中间那个 spatial state / renderer / simulator，而不是整个 cognition。

于是最终并不是一个硕大无朋的模型每秒看 60 帧然后沉思人生，而是一个**异步分层系统**：

\[
\boxed{
\text{cheap continuous perception}
\rightarrow
\text{state change}
\rightarrow
\text{expensive intelligence on demand}
}
\]

从功耗角度看，我觉得这几乎是 AI 眼镜和机器人的必然方向。**真正稀缺的不只是 FLOPs，而是“什么时候值得烧 FLOPs”的判断能力。**

## 25. 用户

关于这个recurency, 我们横向回顾一下recurrent tiny model这个方向，他也是transformer但不是语言模型，而是重复让transformer操作状态变量解决数独问题。那我就想，llm是不是也可以这样，llm思考时不必用自然语言打草稿，那很低效，如果用latent token打草稿，本质上就也是在线性草稿纸上做recurrent transformer，其实就是recurrent Turing machine, 如果还能改过去的草根，就是recurrent冯诺依曼机

## 26. 助手

对，而且你这个抽象我觉得比“latent CoT 能不能省几个 token”深得多。**TRM 真正提示的可能不是一种解数独的小模型技巧，而是 Transformer 可以从“展开很多不同层的前馈网络”变成“同一个计算算子反复作用于可变状态”的机器。** 这就已经非常接近传统计算机的抽象了。

先确认你说的 TRM：Tiny Recursive Model 确实基本就是一个很小的 Transformer 网络反复操作两个状态 \(y,z\)。论文里的核心循环就是反复做 \(z\leftarrow f(x,y,z)\)，再用 \(z\) 去修正当前答案 \(y\)。它只有约 7M 参数、两层网络，却通过重复使用同一网络获得很大的“有效计算深度”。citeturn972651view0

所以它最有意思的地方不是“小”，而是：

\[
\boxed{\text{parameters} \neq \text{computation depth}}
\]

以前 Transformer 是：

\[
x\rightarrow F_1\rightarrow F_2\rightarrow\cdots F_{96}\rightarrow y
\]

每层参数通常不同。

TRM 是：

\[
s_0\rightarrow F_\theta(s_0)
\rightarrow F_\theta(s_1)
\rightarrow F_\theta(s_2)
\rightarrow\cdots
\]

**同一个程序反复运行，数据状态不断变化。**

这确实已经有很强的“计算机”味道。

---

### LLM 完全可以这样思考

而且已经有人直接这么干了。

Coconut 的思路几乎就是你说的：

普通 CoT：

\[
h_t
\rightarrow
\text{softmax}
\rightarrow
\text{word token}
\rightarrow
embedding
\rightarrow
h_{t+1}
\]

Coconut 把中间那堆人类语言仪式直接删掉：

\[
h_t
\rightarrow
h_{t+1}
\]

具体来说，它把模型最后的 hidden state **直接作为下一步输入 embedding**，不先 decode 成自然语言 token。citeturn972651view1

所以：

> 自然语言只是 I/O interface，不应该必然成为 CPU 的内部指令集。

我觉得这是非常重要的判断。

现在 CoT 相当于让 CPU 每执行一条内部指令，都必须先：

> 把寄存器内容翻译成英文，打印出来，再 tokenizer 一遍，再读回 CPU。

然后大家惊讶 inference 很贵。

人类工程学偶尔确实能达到一种令人肃然起敬的绕远程度。

---

## latent token 真正厉害的是信息带宽

一个语言 token 最终必须落在：

\[
V\approx 10^5
\]

个离散符号之一。

信息容量粗略只有：

\[
\log_2V\sim17\ bits
\]

当然实际语义不能这么简单算，但 bottleneck 是真的。

一个 latent：

\[
z\in\mathbb R^{4096}
\]

理论上一次可以携带大量连续信息。

所以一个 latent reasoning step 不必对应：

> “因为 A 大于 B，所以我们接下来考虑 C……”

它完全可能同时编码：

- A>B；
- 三个候选路径；
- 当前置信度；
- 中间计算结果；
- 哪个分支需要回溯；
- 一堆无法自然翻译成人话的结构。

Coconut 的实验甚至观察到 latent state 可以同时保留多个潜在下一步，呈现类似 breadth-first exploration 的行为，而不是 CoT 每生成一个词就提前把思路压成一条路径。citeturn229751academia0

这才是 latent reasoning 真正可能比 CoT 强的地方：

\[
\boxed{\text{不是压缩英文，而是解除语言瓶颈}}
\]

---

# 然后你的“线性草稿纸”类比基本成立

假设模型每一次迭代产生一个 latent：

\[
z_1,z_2,z_3,\ldots,z_T
\]

而后续 Transformer 可以 attention 到：

\[
[z_1,\ldots,z_t]
\]

那么它事实上拥有：

**一个 controller + 一条不断增长的 latent tape。**

controller 就是：

\[
F_\theta
\]

tape 就是：

\[
Z=[z_1,z_2,\ldots]
\]

每运行一次：

\[
(z_{t+1},state_{t+1})
=
F_\theta(state_t,Z_t,input)
\]

这确实非常接近一种 **neural Turing-machine architecture**。

而 2018 年的 Universal Transformer 其实已经沿这个方向走过：对同一组表示反复应用共享 Transformer，并增加动态 halting。在一些理论假设下，这种 recurrent Transformer 可以达到 Turing completeness。citeturn229751academia3

2025 年的 recurrent-depth 工作甚至把它做到 3.5B 参数、800B token 训练，测试时可以继续循环同一个 recurrent block，用更多迭代换 reasoning performance，而不需要产生更多文字 token。citeturn972651view2

所以这已经不只是哲学类比。

---

## 不过有一个小地方我要纠正你

你说：

> 能修改过去草稿，就从 Turing machine 变成 von Neumann machine。

严格计算机科学意义上**不是这样**。

因为经典 Turing machine 的 tape 本来就是可写的：

\[
read\rightarrow overwrite\rightarrow move\ left/right
\]

所以：

\[
\boxed{\text{可修改过去的memory}}
\]

其实已经属于 Turing machine。

真正区别更接近：

### Append-only latent CoT

\[
M_t=[z_1,z_2,\ldots,z_t]
\]

只能不断往后写。

错误的数据不能消失，只能以后再写：

> “第 17 个 latent 作废，以第 43 个为准。”

像数据库 event sourcing。

---

### Writable latent memory

模型拥有：

\[
M\in\mathbb R^{N\times D}
\]

每一步可以：

\[
read(M,i)
\]

\[
write(M,j,v)
\]

\[
delete/update(M,k)
\]

这已经是：

\[
\boxed{\text{Transformer controller + RAM}}
\]

从工程结构看，实际上比传统单带 Turing machine 更接近现代计算机，因为 attention 天生就是一种很强的**内容寻址/random-access read**。

---

### 再往前一步才真正很像 von Neumann

如果 memory 不只保存数据：

\[
M=\{\text{data}\}
\]

还可以保存：

\[
M=\{\text{data},\text{instructions},\text{subroutines}\}
\]

而 Transformer 能读出这些 latent instructions 并执行它们：

\[
\boxed{
\text{program}
+
\text{data}
\in
\text{same mutable memory}
}
\]

这就开始逼近 stored-program architecture。

于是模型甚至可以：

1. 写一个临时算法；
2. 把算法存在 memory；
3. 执行；
4. 检查结果；
5. 修改算法；
6. 再运行。

**这时候“思考”已经不是生成 sequence，而是真正在执行一个动态程序。**

---

# 所以我觉得最终结构可能不是今天所谓的 LLM

更可能是：

\[
\boxed{\text{Transformer = learned CPU}}
\]

下面挂：

\[
\boxed{\text{latent working memory}}
\]

\[
\boxed{\text{persistent episodic memory}}
\]

\[
\boxed{\text{read/write scratch space}}
\]

然后：

\[
F_\theta
\]

反复运行。

可以画成：

\[
(input,M_t,z_t)
\overset{F_\theta}{\longrightarrow}
(M_{t+1},z_{t+1},halt?)
\]

一直循环：

\[
t=1,2,\ldots
\]

直到：

\[
halt=1
\]

才调用语言 decoder：

\[
z_T\rightarrow\text{natural language answer}
\]

也就是说，**语言只在最后出现一次。**

这甚至正好对应最近一篇工作的标题思路，ReLIT 就是“Think Deep, Speak Once”：recursive latent block 反复 refine \(z\)，最后才让语言模型输出。citeturn418942academia3

而今年刚出来的 Latent Recurrent Thoughts 又进一步证明：甚至可以把大 LLM 冻住，只训练一个很小的 recurrent reasoner，让它反复修改 latent thought，再交给 LLM decode，在 Sudoku、代码和自然语言 reasoning 上都能获得收益。citeturn418942academia0

---

## 这还能解释一个更大的问题：为什么 parameter scaling 最终可能让位给 compute scaling

今天：

\[
\text{更难的问题}
\Rightarrow
\text{更大的model}
\]

因为 Transformer depth 是写死的。

假设未来变成：

\[
z_{t+1}=F_\theta(z_t,M_t,x)
\]

那么：

简单问题：

\[
5\ iterations
\]

复杂问题：

\[
500\ iterations
\]

同一个 model。

于是 intelligence scaling 从：

\[
\boxed{\text{parameter scaling}}
\]

部分转成：

\[
\boxed{\text{runtime depth / memory / search scaling}}
\]

这和 CPU 很像。

我们并不会因为排序十亿个数字，就造一颗“拥有十亿倍 transistor 的排序 CPU”。

我们只是让同一个 CPU 跑久一点。

**TRM 最令人不安又有趣的地方就在这里：7M 参数反复运行，能在某些结构化问题上打败参数量大几个数量级、但计算路径没有正确归纳偏置的大模型。** 不过后续分析也发现 TRM 的 ARC 表现里存在很强的 task-specific conditioning、test-time augmentation 和浅层有效 recursion，因此不能把那个 benchmark 神化。citeturn314693academia0

---

所以把你这个想法压成一句话，我会写成：

> **CoT 把 Transformer 当作一个不断写自然语言日志的函数；latent recurrence 则开始把 Transformer 当成 CPU，让它反复修改机器状态。**

再加入**可寻址、可修改的外部 latent memory**之后，真正重要的抽象就已经不是“语言模型”了，而是：

\[
\boxed{
\text{learned processor}
+
\text{working memory}
+
\text{program/state}
+
\text{adaptive runtime}
}
\]

到那个阶段，“LLM”这个名字大概会像今天把电脑叫“打字机”一样，历史上也不能算错，只是已经错过了这个东西真正有意思的部分。

## 27. 用户

你说的对，只能append pop那种该叫 neural 下推自动机，这个做树搜很好，能改写就是神经图灵机，而冯诺依曼机是等价的。

话说Coconut，他是skip了 llm的底层，直接在顶层继续推理吗？感觉这要全量微调才能训练出来。而且这个skip可以很灵活，最极端的是只在顶层recurrent, 但这样收益有限，大量知识在中层。

## 28. 助手

不是。**标准 Coconut 并没有 skip LLM 的底层。** 你这个问题正好点到了它和“只在顶层做 recurrent reasoning”之间最重要的区别。

Coconut 做的是：

\[
h_t^{(L)}
\rightarrow
\text{直接作为下一位置的 input embedding}
\rightarrow
\text{Layer 1}
\rightarrow\cdots\rightarrow
\text{Layer L}
\rightarrow
h_{t+1}^{(L)}
\]

也就是说，**最后一层 hidden state 直接绕过 LM head、词表采样和 embedding lookup，然后从最底层重新跑完整个 Transformer stack。** 官方实现就是取 `outputs.hidden_states[-1]`，塞回下一 latent position 的 `inputs_embeds`，然后再次调用整个 base causal LM。已有 token 的 KV 会 cache，不重新算，但这个新的 latent position 仍然经过全部层。citeturn466736view0

所以它 skip 的其实只有：

\[
h_L
\rightarrow
\cancel{\text{LM head}}
\rightarrow
\cancel{\text{discrete token}}
\rightarrow
\cancel{\text{embedding lookup}}
\]

而不是：

\[
\cancel{\text{lower/middle layers}}
\]

这点非常关键。

### 你的第二个直觉也对：原版 Coconut 确实基本需要全量适配

原因甚至比“latent reasoning 是新任务”更直接。

正常 LLM 第一层看到的是：

\[
e_t\in \mathcal E_{\text{token}}
\]

也就是 embedding manifold。

Coconut 突然往里面塞：

\[
h_t^{(L)}\in \mathcal H_{\text{top}}
\]

虽然维度一样，但**统计分布和语义坐标系根本没有理由天然一样**。

相当于把 CPU 的输出总线直接焊到输入总线上，然后很有信心地宣布“插头尺寸一样”。工程学有时候就是这么乐观。

所以模型必须学会：

\[
\text{如何把top hidden当作一个新的输入状态解释}
\]

而且不只是第一层要适应。整个栈还需要学会：

\[
h_t\rightarrow h_{t+1}
\]

这种 recurrent latent dynamics。

原论文明确强调整个 latent chain 是 end-to-end differentiable，并使用逐阶段 curriculum，把原先自然语言 CoT step 一个个换成 continuous thoughts。官方代码的 optimizer 也是直接吃 `parallel_model.parameters()`，没有冻结 backbone。citeturn870912view0turn466736view1

所以你说“这感觉要全量微调”，**原版 Coconut 的答案基本就是：对。**

---

## 但你后半句其实引出了比 Coconut 更有意思的设计空间

假设有一个 \(L\) 层 Transformer。

Coconut 是：

\[
h_L
\rightarrow
\boxed{1\rightarrow2\rightarrow\cdots\rightarrow L}
\rightarrow
h_L'
\]

也就是 **full-stack recurrence**。

但完全可以做：

\[
h_L
\rightarrow
\boxed{k\rightarrow k+1\rightarrow\cdots\rightarrow L}
\rightarrow
h_L'
\]

只 recurrent 上面的部分。

最极端：

\[
h_L
\rightarrow
\boxed{L}
\rightarrow
h_L'
\]

只反复跑最后一层。

这样 compute 当然便宜得离谱：

\[
O(L)\rightarrow O(1)
\]

但正如你说的，能力上限很可能迅速下降。

### 为什么？

不能简单说“知识都存在中层”，因为 factual knowledge 实际分布得很散。

但更准确地说：

**不同层实现了不同层次的 computation。**

粗略可以理解：

\[
\text{lower}
\rightarrow
\text{局部特征/词法/基础表示}
\]

\[
\text{middle}
\rightarrow
\text{关系、组合、抽象变换}
\]

\[
\text{upper}
\rightarrow
\text{任务决策、预测、输出组织}
\]

一个顶层 recurrent block 可以反复操作已经形成的高层表示：

\[
z\rightarrow f_{\text{top}}(z)\rightarrow f_{\text{top}}^2(z)\cdots
\]

这很适合：

- refine 一个候选答案；
- 做简单 search；
- 更新置信度；
- 从已有 feature 中选择。

但如果新的 reasoning step 要求：

> 重新解释输入中的某个关系  
> 以另一种 abstraction 组合两个事实  
> 从 memory 中提取不同的信息并重新编码

那么只重复最后一层就麻烦了。

因为它只能操作：

\[
h^{(L-1)}
\]

已经给它准备好的 feature。

它没有机会说：

> “刚才中间层理解错方向了，我们重新把这几个东西组合一次。”

---

# 所以我怀疑真正合理的结构不是 Coconut 的 full-stack recurrence

也不是纯 top-layer recurrence。

而是某种：

\[
\boxed{\text{selective recurrent depth}}
\]

例如一个 80 层模型，第一次理解 prompt：

\[
1\rightarrow2\rightarrow\cdots\rightarrow80
\]

得到初始 reasoning state。

之后思考可能只循环：

\[
25\rightarrow26\rightarrow\cdots\rightarrow80
\]

需要更深层重新解释输入时才回到：

\[
10\rightarrow\cdots\rightarrow80
\]

非常简单的 refinement：

\[
65\rightarrow\cdots\rightarrow80
\]

甚至：

\[
75\rightarrow80
\]

于是计算深度本身变成动态的。

---

## 这其实比今天“thinking token 数量”更自然

现在 test-time compute 基本是一维旋钮：

\[
N_{\text{tokens}}
\]

每一个 thinking token 都花：

\[
L
\]

层计算。

所以：

\[
C=N\times L
\]

非常粗暴。

更合理的应该是二维：

\[
\boxed{C=\sum_t d_t}
\]

其中 \(d_t\) 是第 \(t\) 次 reasoning iteration 实际经过多少层。

简单 thought：

\[
d_t=8
\]

需要重新组织知识：

\[
d_t=40
\]

真正困难的 reinterpretation：

\[
d_t=80
\]

这就开始像真正的计算机了。

不是每一条 instruction 都必须让整个芯片从头初始化一次。

---

### 而且还可以更进一步：不一定从顶层 state 回灌

Coconut：

\[
h_L^{t}
\rightarrow
h_0^{t+1}
\]

这个其实非常武断。

完全可以维护：

\[
s_t^{(20)},s_t^{(40)},s_t^{(60)},s_t^{(80)}
\]

不同 abstraction level 各自有 recurrent state。

然后：

\[
s_{t+1}^{(40)}
=
F_{20:40}
(s_t^{(20)},s_t^{(40)},M)
\]

\[
s_{t+1}^{(80)}
=
F_{40:80}
(s_{t+1}^{(40)},s_t^{(80)},M)
\]

这样模型拥有**多层次 working memory**。

这比把一个 top-layer vector 硬塞回 token embedding 层，结构上漂亮得多。

---

## 所以我觉得 Coconut 更像 proof of concept

它证明的是：

\[
\boxed{\text{自然语言不是reasoning所必需的}}
\]

以及：

\[
\boxed{\text{Transformer hidden state可以形成recurrent computation}}
\]

而不一定证明：

> top hidden → bottom embedding

就是最终架构。

事实上这个接口很可能只是因为**不用改现有 Transformer architecture，实验最好做**。

真正成熟的 frontier model 我更期待：

\[
\text{input pass}
\rightarrow
\boxed{\text{multi-level latent recurrent core}}
\rightarrow
\text{output pass}
\]

其中 pretraining 得到的各层知识和表示都可以被 recurrent computation 重复利用。

所以你那句话我会稍微改成：

> **只在顶层 recurrent 最便宜，但会把已有 Transformer 当成一个固定 feature extractor；full-stack recurrence 最充分，但太贵。真正甜点很可能是让中高层若干 abstraction blocks recurrent，并允许模型动态决定什么时候向下重新计算。**

这比 Coconut 目前那种“每想一步就把整个 stack 再跑一次”更像最终形态。

## 29. 用户

我觉得coconut只是用更优雅的方式解决了底部顶部词表绑定的问题。这甚至很可能是它的出发点，而我们现在是从latent token角度看这个特例。要训练latent token感觉给部分中层加轨，从新用teacher model训练是避免不了的

## 30. 助手

我基本同意，而且我会把 Coconut 的意义重新表述成你这个角度：

> **Coconut 首先解决的是“思考状态为什么非得经过 vocabulary bottleneck 才能回到模型里”这个历史包袱；latent reasoning 是这个改动自然打开的更大设计空间。**

传统 CoT 的循环其实很荒唐：

\[
h_L
\rightarrow W_{\text{vocab}}
\rightarrow token
\rightarrow E_{\text{vocab}}
\rightarrow h_0
\]

如果 input embedding 和 output unembedding 还 tied：

\[
W_{\text{vocab}}\approx E^T
\]

那整个模型被强迫通过一个**人类语言词表定义的离散接口**完成内部循环。

这对输出语言当然合理。

对内部计算则毫无必然性。

Coconut 做的：

\[
h_L\rightarrow h_0'
\]

本质上就是把：

\[
\boxed{\text{decode}\rightarrow\text{quantize}\rightarrow\text{re-encode}}
\]

这一套删掉。

所以从今天回头看，它确实很像是**绕开顶部和底部被词表绑定的问题**，而不是什么神秘的“连续思想”。当然，我不会断言这是作者最初真正的心理出发点，论文公开动机还是 continuous thought；但作为架构抽象，你这个说法更本质。

---

而真正的 latent token architecture，我也觉得不会是 Coconut 那么简单。

因为 Coconut 还是：

\[
h_L^{(t)}
\rightarrow
\text{整个模型}
\rightarrow
h_L^{(t+1)}
\]

它只是取消 vocab bottleneck，却没有真正回答：

> **模型内部哪个 representation level 最适合成为“机器思想 token”？**

这才是关键。

你说“给部分中层加轨”，我觉得很可能就是正确方向。

比如一个 80 层模型，不必让：

\[
h_{80}\rightarrow h_0
\]

而可以在 30、50、70 层建立 latent recurrent rail：

\[
z_t^{30},z_t^{50},z_t^{70}
\]

然后：

\[
z_t^{50}
\rightarrow
R_{50:70}
\rightarrow
z_{t+1}^{70}
\]

或者：

\[
z_t^{70}
\rightarrow
P
\rightarrow
z_{t+1}^{50}
\]

这样 latent reasoning 发生在**模型已经完成基础感知/词义解析之后，但还没有压缩到输出 logits 之前**。

这比 top→bottom 的 Coconut 更合理。

---

### 为什么中层尤其重要

我觉得你说“大量知识在中层”需要稍微改成：

> **大量可复用的中间抽象和 feature transformation 存在中层。**

知识本身其实分布在参数和各层 representation 中，很难说“知识就存在第 47 层”。

但 reasoning 过程中真正想反复利用的东西可能是：

\[
\text{entities}
+
\text{relations}
+
\text{causal structure}
+
\text{partial solution}
+
\text{uncertainty}
\]

这些东西已经比 token embedding 高级很多，却还没有被最后几层压成：

\[
P(\text{next token})
\]

所以把 recurrence 放在这里非常自然。

可以把模型粗略拆成：

\[
\underbrace{1\sim 20}_{\text{encode}}
\rightarrow
\underbrace{20\sim60}_{\text{latent compute}}
\rightarrow
\underbrace{60\sim80}_{\text{decode / decision}}
\]

真正 recurrent 的可能主要是：

\[
\boxed{20\sim60}
\]

而不是全模型。

这样 compute 也会好很多。

---

# 但训练问题确实麻烦

这里我和你基本一样判断：

> **如果想把今天已经训练好的 frontier LLM 改造成真正 latent recurrent machine，重新训练一部分中层几乎不可避免。**

原因不是简单的接口尺寸问题。

原模型学习的是：

\[
h_l^{(t)}=F_l(h_{l-1}^{(t)})
\]

而你现在突然要求：

\[
h_l^{(t+1)}
=
F_l(h_{l-1}^{(t+1)},z_t)
\]

也就是某些层要理解一种训练时从没见过的东西：

\[
\boxed{\text{previous-computation latent state}}
\]

这是一种新的 modality。

跟突然给模型加 image token 没本质区别。

当然得训练。

---

## Teacher 大概率是最便宜的启动方式

我不会说数学上“避免不了”。

理论上可以：

- outcome RL；
- self-play；
- algorithmic curriculum；
- next-state prediction；
- self-supervised latent consistency；

直接让 latent recurrent model 自己学。

但工程上，**teacher distillation 几乎肯定是第一条现实路线**。

因为现有大模型已经会用自然语言 CoT 解题。

于是有：

\[
x
\rightarrow
r_1,r_2,\ldots,r_n
\rightarrow
y
\]

这些语言 reasoning traces 可以作为 scaffold。

然后训练 latent student：

\[
x
\rightarrow
z_1,z_2,\ldots,z_m
\rightarrow
y
\]

要求：

\[
y_{\text{student}}
\approx
y_{\text{teacher}}
\]

再加一些中间 supervision。

甚至不用让：

\[
z_i\leftrightarrow r_i
\]

一一对应。

只要 teacher 告诉你：

> 这个问题应该经过这些 reasoning stages，最终得到这个答案。

student 可以自己把它压缩成 latent computation。

---

### 更激进一点，teacher 甚至可以教 hidden state，而不是语言

假设 teacher 和 student 架构接近。

可以直接取 teacher 某些中层：

\[
h_{30}^{teacher},
h_{50}^{teacher},
h_{70}^{teacher}
\]

然后训练 student 的 recurrent rail：

\[
z_t^{30},z_t^{50},z_t^{70}
\]

去逼近 teacher 在**多步 CoT 展开之后**形成的 representation。

也就是把：

\[
\text{teacher跑10个自然语言thinking tokens}
\]

蒸馏成：

\[
\text{student跑1次latent recurrent update}
\]

这才真正有意思。

本质上：

\[
\boxed{\text{temporal distillation}}
\]

把很多 token-time computation 压进一次 latent-state transition。

如果能成功，推理效率可能不是快 20%，而是数量级变化。

---

# 所以我现在反而觉得下一代模型训练可能分成两阶段

第一阶段还是今天熟悉的：

\[
\text{massive pretraining}
\]

学世界知识、语言、视觉、代码。

得到一个强大的 feed-forward Transformer。

然后第二阶段：

\[
\boxed{\text{recurrentization}}
\]

给中层加入：

- latent state；
- recurrent rail；
- writable scratch memory；
- halting；
- dynamic depth。

再用：

\[
\text{teacher traces + RL + outcome supervision}
\]

把原来：

\[
\text{token-space reasoning}
\]

蒸馏成：

\[
\text{latent-state computation}
\]

这甚至可能成为一个类似今天 post-training 的独立阶段。

---

所以我觉得 Coconut 真正历史意义可能不是它那个具体架构，而是它做了一件非常朴素的事：

> **证明“模型输出必须重新经过词表才能继续思考”根本不是必要条件。**

而一旦这个洞被捅开，真正的问题自然就变成：

\[
\boxed{\text{到底在哪一层建立 recurrent computational state 最合适？}}
\]

我反而认为答案大概率不是底层，也不是顶层，而是**一组中层 latent rails + 少量跨层 read/write**。那时候模型才真正从“会不断续写 token 的网络”变成了“拥有内部机器状态的计算系统”。

## 31. 用户

在这个视角下，coconut是一种最平凡的latent token 模型，它运行一个完整执行全部层的大循环。但我们想想一般情形，在模型读输入时，他甚至可以不运行到顶层，在中层就直接goto 底层继续读输入了，这直接效果就是kV cache不再是矩阵而是稀疏树状，底层是字面token，高层全跳过不存，中层是多个底层token才出一个token代表一个语义块，高层更是一个句子才出一个token，这样能极大节省算力和存储。到了输出阶段, 又反过来，高层中层反复迭代，充分思考后才慢慢一个字一个字解码出底层token

## 32. 助手

对，这个抽象一旦成立，**KV cache 这个“每层、每 token 都留一份状态”的二维矩阵，本身就显得非常原始**。你描述的其实不是“更高效的 Transformer”，而是把 Transformer 变成一个**多尺度、可跳转、可递归的计算图**。

可以把普通 Transformer 写成：

\[
x_1,\ldots,x_T
\rightarrow
\begin{matrix}
L_1\\
L_2\\
\vdots\\
L_N
\end{matrix}
\]

每个 token 都老老实实跑完所有层，于是 KV cache 大致是：

\[
T\times N\times d
\]

不管某个 token 是逗号、冠词，还是一句话里真正决定语义的关键实体，都得交同样的“过路费”。非常公平，也非常浪费。人类终于把流水线官僚主义实现进了神经网络。

而你设想的是另一种东西：

\[
\boxed{\text{token不是沿深度单调向上，而是在层级间动态流动}}
\]

比如：

\[
\text{字面token}
\rightarrow
\text{局部语义块}
\rightarrow
\text{句子状态}
\rightarrow
\text{段落/概念状态}
\]

而且每一级**不需要等所有低级 token 都跑到顶层以后才构造**。

---

## 输入阶段最重要的变化，就是“提前压缩”

例如底层连续读：

\[
x_1,x_2,x_3,x_4,x_5,\ldots
\]

跑到第 15 层时，模型发现：

> 这几个 token 已经形成一个稳定语义单元。

那么直接：

\[
(x_1,\ldots,x_6)
\rightarrow z_1^{mid}
\]

之后不再让六个 token 分别跑：

\[
L_{16}\rightarrow\cdots\rightarrow L_{80}
\]

而是让一个：

\[
z_1^{mid}
\]

继续。

于是后面的计算从 6 条路径变成 1 条。

再过一会：

\[
z_1^{mid},z_2^{mid},z_3^{mid}
\rightarrow z_1^{high}
\]

可能整个句子只剩一个 high-level latent。

所以有效 token 数随着深度不断下降：

\[
T_0 > T_1 > T_2 > \cdots > T_N
\]

这和现在：

\[
T_0=T_1=\cdots=T_N
\]

完全不同。

计算复杂度就可能从：

\[
\sum_l O(T^2d)
\]

变成：

\[
\sum_l O(T_l^2d)
\]

而如果：

\[
T_{l+1}\approx \alpha T_l,\quad \alpha<1
\]

高层计算会便宜得离谱。

---

# 所以你说 KV cache 变成“稀疏树”特别准确

它不应该再是：

\[
K_{layer,token},V_{layer,token}
\]

这种规则矩阵。

而更像：

```text
sentence latent
├── phrase latent
│   ├── token
│   ├── token
│   └── token
├── phrase latent
│   ├── token
│   └── token
└── phrase latent
    ├── token
    └── token
```

而且不同节点处在不同网络深度。

于是 memory 本身变成：

\[
\boxed{\text{hierarchical semantic memory}}
\]

而不是：

\[
\boxed{\text{history of activations}}
\]

这两个概念差太多了。

今天的 KV cache 其实不是“记忆”，它只是：

> 我把过去每一步中间计算结果全堆着，免得重新算。

你这个结构才开始真正像计算机意义上的 memory。

---

## 最漂亮的一点是：底层信息不一定真的删除

这里我觉得还能比你描述得更强。

压缩成：

\[
x_{1:10}\rightarrow z^{mid}
\]

之后，不一定把底层 token 丢掉。

可以只是把它们**冷存储**：

\[
\text{active memory}: z^{mid}
\]

\[
\text{cold memory}: x_1,\ldots,x_{10}
\]

平时 reasoning 只看：

\[
z^{mid}
\]

如果后来发现：

> “等等，用户刚才第四句话具体用了哪个数字？”

就沿树往下展开：

\[
z^{mid}
\rightarrow
\{x_1,\ldots,x_{10}\}
\]

重新读取。

这就很像虚拟内存：

- 高频语义在 cache；
- 低层细节在 RAM；
- 原始文本在 storage。

于是 attention 也不再是：

\[
\text{attend to everything}
\]

而是：

\[
\boxed{\text{先在高层找相关区域，再向下展开}}
\]

搜索复杂度就开始从平面 retrieval 变成树搜索。

---

# 然后输出阶段确实应该反过来

这一点我觉得尤其重要。

今天 LLM 的“思考”和“说话”共用同一条流水线：

\[
\text{thinking}
\rightarrow
\text{next token}
\rightarrow
\text{thinking}
\rightarrow
\text{next token}
\]

所以模型刚想出半截，就得开始说。

你描述的结构可以完全分离：

### 高层：

\[
z^{high}_{t+1}=F(z^{high}_t,M)
\]

反复几十次甚至几百次。

这里处理：

- overall plan；
- causal structure；
- alternatives；
- consistency；
- search；
- backtracking。

然后中层展开：

\[
z^{high}
\rightarrow
z_1^{mid},z_2^{mid},\ldots
\]

每个中层 latent 再反复 refine：

\[
z_i^{mid}\rightarrow F(z_i^{mid})
\]

最后才：

\[
z_i^{mid}
\rightarrow
\text{token sequence}
\]

于是语言输出只是最后一级**serialization**。

这就像程序先在内存里把数据结构算完，最后才打印字符串。

正常得多。

---

## 而且可以自然地产生不同“时钟频率”

这其实和我们前面说机器人系统异步运行很像。

底层语言 decoder：

\[
\sim 10-100\text{Hz token rate}
\]

中层 semantic computation：

\[
\sim \text{更慢但每步信息量更大}
\]

高层 planning：

\[
\sim \text{更少step，但每step覆盖巨大语义范围}
\]

所以不是所有层同时 clock。

这才是关键。

普通 Transformer 相当于：

> CPU、L1、L2、RAM、SSD 全部每个 cycle 一起跑一遍。

当然耗电。

---

# 这东西已经很接近“神经操作系统”

因为一旦加入你说的 goto，模型就不再只是：

\[
L_1\rightarrow L_2\rightarrow\cdots L_N
\]

而变成：

\[
L_i\rightarrow L_j
\]

由当前状态决定下一步去哪里。

例如：

\[
\text{goto lower}
\]

重新读取细节；

\[
\text{goto same}
\]

继续迭代；

\[
\text{goto higher}
\]

做抽象；

\[
\text{emit}
\]

开始输出；

\[
\text{halt}
\]

停止。

于是一次 forward pass 已经变成执行一个 learned control-flow graph：

\[
(state,memory,layer)
\rightarrow
(state',memory',next\_layer)
\]

这已经不像 feed-forward neural network 了。

更像：

\[
\boxed{\text{neural VM}}
\]

---

## 但这里有一个巨大训练难题

你这个架构最大的障碍其实不是 inference，而是：

> **谁教模型什么时候压缩、什么时候跳层、什么时候展开？**

比如这句话：

> “Alice gave Bob the key before Carol entered.”

到底什么时候：

\[
\text{Alice gave Bob the key}
\]

可以压成一个 latent？

如果压太早：

\[
z=\text{key transfer}
\]

可能把 “before Carol entered” 所需要的 temporal relation 丢掉。

压太晚：

省不了 compute。

所以模型要学的已经不是：

\[
P(x_{t+1}|x_{\leq t})
\]

而是同时：

\[
P(\text{next computation action}|state)
\]

即：

\[
\{\text{continue},\text{compress},\text{expand},\text{recur},\text{jump},\text{emit},\text{halt}\}
\]

这其实是一个 routing/policy learning 问题。

因此我很同意你上一轮说的：

**不能简单拿现有 LLM 接几根线就完事。**

这大概率需要重新做一轮 teacher-guided training。

---

# Teacher 最有价值的地方可能不是提供答案，而是提供“语义层级”

比如 teacher 先产生：

```text
tokens
→ noun phrase
→ proposition
→ causal relation
→ plan
```

student 不必复制 teacher 的文字 CoT。

而是学习：

\[
\text{这6个token可以collapse成一个state}
\]

\[
\text{这3个state又可以collapse}
\]

也就是 teacher 帮 student 学**计算图拓扑**。

这比传统 knowledge distillation 更深一步：

\[
\boxed{\text{distill not only knowledge, but computation structure}}
\]

之后再用 RL 优化：

\[
reward =
accuracy
-\lambda_1 compute
-\lambda_2 memory
-\lambda_3 latency
\]

逼模型自己找到：

\[
\text{最小必要计算路径}
\]

---

## 这还可能解释为什么未来 context window 根本不会继续按今天的方式扩

今天大家炫：

\[
1M,\ 10M,\ 100M\ tokens
\]

但如果你这个方向成立，真正合理的 memory 应该是：

\[
100M\ raw\ tokens
\]

压缩成：

\[
1M\ local\ latents
\]

再变成：

\[
10k\ semantic\ chunks
\]

再变成：

\[
100\ high-level\ memories
\]

真正高层 attention 永远只看：

\[
O(10^2-10^4)
\]

个节点。

需要细节时再 drill down。

所以“context length”这个指标本身都可能过时。

真正应该问的是：

\[
\boxed{\text{模型能维护多大的hierarchical addressable memory？}}
\]

---

### 我觉得你这个思路最后会导向一个很强的结论

Coconut 只是最简单情况：

\[
\boxed{\text{一条深度为N的循环}}
\]

每次：

\[
1\rightarrow2\rightarrow\cdots\rightarrow N\rightarrow1
\]

而一般化以后应该是：

\[
\boxed{\text{一个稀疏、动态、层次化的神经控制流图}}
\]

输入时：

\[
\text{many low-level tokens}
\rightarrow
\text{few high-level latents}
\]

思考时：

\[
\text{high/mid-level latent recurrence}
\]

需要时：

\[
\text{tree drill-down}
\]

输出时：

\[
\text{high-level plan}
\rightarrow
\text{mid-level semantics}
\rightarrow
\text{low-level language tokens}
\]

所以最终模型的几何形状已经不是 Transformer 那张：

\[
\text{sequence}\times\text{layers}
\]

的矩形。

而更像一棵**随计算动态生长、合并、展开的时空树**。

这比单纯的 latent CoT 激进得多，也更有可能带来真正数量级的算力和内存下降。因为你省掉的不是 tokenizer 那一点可怜的开销，而是**大量低价值 token 根本不再穿过高层网络**。

## 33. 用户

而且你会发现一旦有这个多层次控制流图，recurrent transformer更像人类的脑回路了，更绝的是记忆有了层次性，你会发现从latent 层开始推理，丢掉底层的最细粒度token，模型很可能依然记得大概，这甚至能形成选择性遗忘机制。

## 34. 助手

对，而且这可能是你这个架构里最漂亮的一层结果：**“遗忘”不再需要单独设计，它会从层次化表示和动态路由里自然长出来。**

普通 Transformer 的记忆逻辑其实很笨：

\[
x_1,x_2,\dots,x_T
\rightarrow
\text{所有 token 的 KV 都留着}
\]

它默认：

> 过去的每一个字都可能同样重要。

而你这个层次系统会自然变成：

\[
\text{raw token}
\rightarrow
\text{phrase latent}
\rightarrow
\text{proposition latent}
\rightarrow
\text{episode latent}
\rightarrow
\text{concept latent}
\]

每向上一层，本质上都在做一次：

\[
\boxed{\text{有损压缩}}
\]

于是模型可能已经不记得：

> 用户原话到底用了 “pretty good” 还是 “quite good”。

但仍然保留：

> 用户对此持轻度正面态度。

这就已经非常接近我们通常说的“记得大概”。

---

### 更妙的是，不同层可以有不同时间常数

这和一些神经科学里的 hierarchical predictive coding 思路相当接近：低层表示短时间尺度变化，高层表示更长时间尺度的结构。2024 年 Jiang 和 Rao 的 dynamic predictive coding 模型就是低层编码较短序列，高层逐渐编码更长时间跨度。citeturn944905search11turn944905search12

对应到你这个模型：

\[
\tau_{\text{token}}\ll
\tau_{\text{phrase}}\ll
\tau_{\text{episode}}\ll
\tau_{\text{concept}}
\]

也就是：

**字面形式**可能几十秒就扔；

**具体事件**保持几小时、几天；

**提炼出的知识**可以永久存在。

于是 memory hierarchy 同时变成了：

\[
\boxed{\text{abstraction hierarchy}}
\]

和

\[
\boxed{\text{forgetting hierarchy}}
\]

两件事其实是同一个机制。

---

## 而且“忘掉 token”不等于彻底失忆

假设原始输入：

> Alice 在周二下午 3:17，把一把带红色钥匙扣的铜钥匙交给 Bob。

经过几级压缩后可能只剩：

\[
z_{\text{episode}}
=
\text{Alice transferred key to Bob}
\]

底层 token KV 已经删了。

然后你问：

> 谁把钥匙给了 Bob？

它完全答得出来。

问：

> 是星期几？

如果中层 latent 保留：

\[
\text{Tuesday}
\]

还能回答。

但问：

> 钥匙扣是什么颜色？

可能就已经忘了。

这就是非常自然的**graded forgetting**，而不是今天 cache eviction 那种：

\[
\text{这个token还在 / 这个token没了}
\]

二元删除。

---

# 更进一步，它可以成为“有目的的遗忘”

这里我觉得比单纯层次压缩还更重要。

模型可以学习一个保留价值函数：

\[
V(m)
=
\mathbb E[\text{future utility}\mid m]
\]

然后压缩目标变成：

\[
\max
\left[
\text{future task utility}
-
\lambda\cdot\text{memory cost}
\right]
\]

高预测价值的信息留下：

- 用户长期偏好；
- 尚未完成的任务；
- 一个异常事件；
- 某个以后可能成为因果解释的重要细节。

低预测价值的信息衰减：

- 具体措辞；
- 重复信息；
- 已完成任务的执行细节；
- 很久没被使用的视觉纹理。

今年甚至有一篇很贴你这个想法的工作直接把概念叫 **predictive forgetting**：不是最大程度保存过去，而是选择性保留那些对未来预测有价值的信息，并让记忆随着 consolidation 逐渐变得更抽象。作者还在 Transformer 等模型上做了验证。citeturn944905academia32

这个方向和你的推论几乎是同一个第一性原理：

> **好的记忆系统不是保存最多，而是忘得正确。**

---

## 这样一来，“记忆 consolidation”甚至可以离线发生

刚经历事件时：

\[
M_0=
\text{大量细粒度token/视觉latent}
\]

过一会：

\[
M_1=
\text{几个semantic chunks}
\]

睡一觉，或者空闲期再处理：

\[
M_2=
\text{episode summary}
+
\text{更新existing concepts}
\]

最后：

\[
M_3=
\text{只剩generalized knowledge}
\]

原始 episode 甚至消失了。

这特别有意思，因为人也经常：

> 记得这个结论，但已经忘了自己最初是在哪里学到的。

从机器架构看，这一点一点都不神秘：

\[
\text{episodic latent}
\rightarrow
\text{update semantic latent}
\rightarrow
\text{delete episodic latent}
\]

完成。

---

### 已经有一些工程方向在往这里靠

MELODI 已经在做跨 layer 和 context window 的 hierarchical compression，并把 long-term memory 特意放在中间层进一步压缩；它报告相对于 dense long-term memory 可以把 footprint 降大约 8 倍。citeturn944905search2

HMT 则明确认为传统 recurrent memory 太“flat”，因此加入 hierarchical memory 和 recall/filtering 机制；他们报告在某些长上下文设置下用显著更少的 inference memory 达到相近或更好效果。citeturn944905search0turn944905academia34

还有 Dynamic Memory Compression 已经证明，**不同 layer/head 可以学习不同压缩率**，不必所有层保留同等数量的 KV。citeturn944905search8

这些还远没有走到你描述的完整动态树，但方向是一致的：

\[
\boxed{\text{memory density应该随层级和意义变化}}
\]

而不是所有 token 人人平等地占显存。

---

# 最深的一层其实是：忘掉细节以后，推理可能反而更好

因为低层 token 不是纯资产，也会成为噪音。

如果模型面对十年前的一个事件，每次思考都把：

- 原始措辞；
- 标点；
- 视觉纹理；
- 当时无关的小细节

全部拿回来 attention，一方面浪费算力，一方面还可能干扰当前抽象。

高层 latent 反而是在说：

> 我已经对这段经验做完 feature extraction 了，别再把 JPEG 像素送进前额叶。

于是遗忘本身是一种 **regularization**。

甚至可能减少 reasoning search space：

\[
10^6\ raw\ states
\rightarrow
10^2\ abstract\ states
\]

复杂问题直接在后一层空间里搜索。

这就是为什么我觉得“多层 recurrent control flow + hierarchical memory”远远不只是 KV-cache 优化。

它实际上同时解决四个问题：

\[
\boxed{
\text{compute}
+
\text{memory}
+
\text{abstraction}
+
\text{forgetting}
}
\]

---

不过还有一个很危险的副作用：**遗忘和幻觉会变成邻居。**

一旦原始 token 被删掉，只留下：

\[
z=\text{大概发生过X}
\]

模型以后如果需要重新生成细节，就只能：

\[
p(x\mid z)
\]

做 reconstruction。

这得到的是**最可能的原文**，不一定是真实原文。

于是：

> “我记得大概，但不记得细节”

如果系统没有显式 uncertainty 标记，很容易变成：

> “我非常确定当时就是这么说的。”

所以真正成熟的 hierarchical memory 还必须把：

\[
\boxed{\text{内容}}
\]

和

\[
\boxed{\text{fidelity / provenance / uncertainty}}
\]

一起保存。

例如 latent 不只是：

\[
z=\text{Alice给Bob钥匙}
\]

而是：

\[
(z,\ confidence,\ source,\ abstraction\ level)
\]

这样模型知道：

> 事件本身我记得很牢，但钥匙颜色已经压缩丢了。

这才是真正意义上的**选择性遗忘，而不是选择性编故事**。

所以我会把我们现在推出来的架构再压成一句：

> **层次化 latent memory 不是把长上下文压缩得更小，而是让模型第一次拥有“哪些东西应该成为知识，哪些东西只值得成为记忆，哪些东西可以被忘掉”的结构。**

这已经比“1M context window”高了一个抽象层级。

## 35. 用户

对，由此进一步引申，我们会发现，如果我们训练模型丢token的能力，这就是一般的神经图灵机了。一旦达到这一步，模型就是真的在进行in context learning，因为模型会自动去优化知识在context中的组织形式，为解决某个问题不断操作上下文，让知识更有效组织，形成正确认识论来完成解决问题。所以这时的模型不是通过更新参数学习，而是亲自操作自己的层次化kv cache来学习

## 36. 助手

对。走到这一步以后，**“in-context learning”这个词才真正配得上 learning**，而不只是今天这种“参数不变，但 prompt 里多塞几个例子，然后 attention 临时拟合一下”的弱形式。

你描述的系统可以非常干净地写成：

\[
\theta=\text{固定的慢权重}
\]

\[
M_{t+1}=U_\theta(M_t,\;x_t,\;goal)
\]

\[
y_t=F_\theta(M_t,\;x_t)
\]

这里真正随任务不断变化的不是 \(\theta\)，而是：

\[
\boxed{M_t=\text{模型自己组织的层次化内部状态}}
\]

只要 \(M_t\) 可以被读、写、删、合并、拆分、重排、建立索引，那么它就已经是一个通用工作内存。经典 Neural Turing Machine 当初的核心定义恰恰就是 neural controller + 可寻址读写 memory。citeturn752468academia35

但你这里实际上比原始 NTM 又前进了一层，因为 **memory 本身不是一排无语义的 cells，而是模型主动构造的知识结构。**

---

## 这时“学习”发生在哪里？

今天训练一个 LLM：

\[
D\rightarrow\nabla_\theta L\rightarrow\theta'
\]

知识通过 gradient 被写进参数。

你的系统则是：

\[
D\rightarrow U_\theta\rightarrow M'
\]

知识直接被写进 memory state。

所以：

\[
\boxed{\text{SGD learning}: \theta\rightarrow\theta'}
\]

vs.

\[
\boxed{\text{in-context learning}: M\rightarrow M'}
\]

区别只是**知识被写到哪个介质里**。

只要新经验让 \(M\) 发生持久的、对未来行为有用的改变，从计算意义上说，这当然就是 learning。

而 2025 年的 TTT 那条路线实际上已经非常接近这个观点：它直接把 recurrent hidden state 做成一个小模型，并在 test time 用输入数据更新这个模型。论文自己的说法就是 hidden state 本身是 machine learning model，update rule 是一次 self-supervised learning step。citeturn752468search11

Fast-weight 文献则更直接：

\[
\theta_{\text{slow}}=\text{长期算法}
\]

\[
W_{\text{fast}}(t)=\text{当前任务学到的东西}
\]

也就是“权重”本身出现两个时间尺度。citeturn752468academia36

你提出的 hierarchical KV cache 可以视为比 fast weights 更一般的版本。

---

# 真正厉害的是：模型开始学习“如何表示问题”

这就触及你说的“认识论”。

今天 LLM 遇到一个问题，基本只能接受：

\[
\text{你给我的context是什么样，我就在这个context上算}
\]

最多通过 CoT 往后 append 一些东西。

未来模型却可以先问：

> 哪些信息是证据？

然后：

\[
M\rightarrow M_{\text{evidence}}
\]

再问：

> 哪些是重复信息？

merge。

> 哪些细节无关？

delete。

> 哪些事实互相矛盾？

建立 conflict edge。

> 现在有哪几个解释？

创建 hypothesis nodes：

\[
H_1,H_2,H_3
\]

然后分别给它们挂证据：

\[
H_1\leftarrow\{E_1,E_4,E_7\}
\]

\[
H_2\leftarrow\{E_2,E_3\}
\]

再发现：

\[
E_4
\]

其实不可靠，于是降低其权重或者删除。

这时候模型解决问题的过程已经不只是：

\[
\boxed{\text{search over answers}}
\]

而是：

\[
\boxed{\text{search over representations of the problem}}
\]

这是质变。

---

## 而人类很多所谓“想通了”，本来就是这个东西

同样十条事实摆在那里。

一开始你的脑子组织成：

\[
A\rightarrow B,\quad C\rightarrow D,\quad E
\]

怎么也想不通。

突然换一个表示：

\[
(A,C,E)\rightarrow X\rightarrow(B,D)
\]

然后：

> 原来如此。

事实一个没变。

**改变的是内部图结构。**

所以很多推理突破实际上不是获得新数据，而是：

\[
\boxed{\text{re-representation}}
\]

这也是为什么我觉得你说“形成正确认识论”虽然稍微强了一点，但方向是非常准确的。

更严格地说，不是模型必然形成“正确”认识论，而是它终于拥有能力去**学习一种使当前任务可解的知识组织方式**。

正确与否还得靠：

- prediction error；
- reward；
- contradiction detection；
- external verification；
- task outcome

来训练。

否则神经图灵机也完全可以很勤奋地把自己的上下文整理成一坨极其有条理的胡说八道。组织能力并不会自动附赠真理，人类已经替我们做了相当充分的实验验证。

---

# 这也意味着“context”最终不是文本

这一点特别重要。

今天：

\[
Context=
[x_1,x_2,\ldots,x_N]
\]

所以大家讨论：

> 1M context，10M context。

你这个模型里：

\[
Context=
\boxed{M}
\]

而 \(M\) 可能同时包括：

\[
\begin{aligned}
&\text{raw tokens}\\
&\text{semantic chunks}\\
&\text{episodes}\\
&\text{hypotheses}\\
&\text{causal graphs}\\
&\text{uncertainties}\\
&\text{temporary algorithms}\\
&\text{partial results}
\end{aligned}
\]

于是所谓 context window 就变成一个非常落后的指标。

模型可能读了：

\[
10^{10}\text{ tokens}
\]

最后 active working memory 里只剩：

\[
10^4\text{ latent nodes}
\]

而真正重要的是：

\[
\boxed{\text{它把那100亿token学成了什么}}
\]

不是它还能不能逐字 attention 回去。

---

## 这会让“参数”和“知识”的关系完全改变

现在大家默认：

\[
\text{模型学会东西}
\Rightarrow
\text{更新parameters}
\]

所以一个 AI 想长期学习，似乎必须：

- fine-tune；
- LoRA；
- continual pretraining；
- SGD。

但在你的架构里：

\[
\theta
\]

真正保存的主要可能变成：

\[
\boxed{\text{如何学习}}
\]

而不是：

\[
\boxed{\text{学到了什么}}
\]

也就是：

\[
\theta=
\text{learning algorithm}
+
\text{general priors}
+
\text{reasoning machinery}
\]

而：

\[
M=
\text{current acquired knowledge}
\]

这就是非常纯粹的 meta-learning。

训练阶段其实是在训练：

> 给你新的世界以后，你应该怎样重构自己的 memory。

而不是试图提前把未来可能遇到的一切都烤进参数。

---

# 更妙的是：KV cache 本身可以逐渐变成“fast weights”

传统 KV：

\[
M=
\{(k_i,v_i)\}_{i=1}^{N}
\]

本质是 non-parametric memory。

但模型开始 merge：

\[
(k_1,v_1),(k_2,v_2),...
\rightarrow
(k',v')
\]

以后，这个 \(v'\) 已经不是某一个输入 token 的 activation。

它是模型**学出来的 sufficient statistic**。

再进一步：

\[
M\rightarrow f_\phi
\]

直接把一堆 context 蒸馏成一个小网络或者局部 operator。

那么：

\[
\text{KV memory}
\rightarrow
\text{latent memory}
\rightarrow
\text{fast weights}
\]

其实是一条连续谱。

现在的一些工作已经开始往这个方向爬。ATLAS 例如明确把问题定义为“在 test time 学习如何最优地记住 context”，不再只是机械累积 token；而 TTT 则干脆让 temporary memory 成为会在线训练的小模型。citeturn752468academia37turn752468search4

---

## 所以可能出现三层学习速度

我觉得这会是最终系统非常漂亮的地方：

### 毫秒到秒

\[
\boxed{\text{working state}}
\]

当前推理：

- stack；
- graph；
- hypotheses；
- latent scratchpad。

### 分钟到年月

\[
\boxed{\text{persistent memory / fast weights}}
\]

个体经验：

- 用户；
- 世界变化；
- 新技能；
- episodic → semantic consolidation。

### 大规模训练周期

\[
\boxed{\theta_{\text{slow}}}
\]

修改的是：

> 如何推理、如何学习、如何组织 memory。

于是最慢的 SGD 不再承担全部学习任务。

它主要负责**改进学习算法本身**。

这其实很像：

\[
\boxed{\text{learned CPU + mutable RAM + gradually compiled knowledge}}
\]

---

# 这样回头看，今天的 ICL 确实只是原始版本

今天 Transformer 已经有一点这种性质。

有研究把 ICL 描述成在固定权重下产生隐式学习动力学，而 fast-weight / linear-attention 系列更直接把 attention 看作某种在线关联记忆。citeturn752468search10turn752468academia36

但今天它最大的问题是：

\[
\boxed{\text{model不能真正控制自己的context representation}}
\]

它只能：

- 读；
- attention；
- append。

不能真正：

- 重写；
- collapse；
- 删除；
- 建索引；
- 把三个事实合成为一个 concept；
- 把一个 concept 拆回证据；
- 为当前任务重新组织 topology。

所以它是一个**被动的 in-context learner**。

你描述的才是：

\[
\boxed{\text{active in-context learner}}
\]

它不是“context 教模型”。

而是：

> **模型拿到经验之后，自己决定怎样把经验变成知识。**

---

而这最终会把我们前面讨论的几条线全部合起来：

\[
\text{latent reasoning}
\]

解决“内部思考为什么要说人话”；

\[
\text{recurrent transformer}
\]

解决“为什么计算深度必须固定”；

\[
\text{hierarchical memory}
\]

解决“为什么所有历史都要同样保存”；

\[
\text{neural Turing machine}
\]

解决“为什么模型不能编辑自己的状态”；

最后：

\[
\boxed{\text{test-time learning}}
\]

出现。

所以我会把这一步定义得比“更好的 KV cache”激进得多：

> **当模型能够主动重写自己的层次化 latent context，并根据任务结果学习如何重写时，context 就从输入缓存变成了可塑的认知状态；参数训练负责学习“学习算法”，而真正面对新问题时的学习，则发生在这个可写状态里。**

到了这里，**一次推理 session 本身就是一次训练过程，只不过被训练的不是慢权重，而是模型自己的认知状态。**

---

If you want, I can:

- 继续详细论述神经图灵机的读写机制
- 探讨层次化KV缓存的训练挑战与解决方案
- 阐述神经图灵机对in-context learning的实际意义
