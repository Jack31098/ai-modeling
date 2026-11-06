# 第 0 步增量 v0.2：独立审核候选、有界流式政策、tokenizer 盘点

本目录接续 `87d01ba`，不覆盖 `../word_boundary_v0/`、旧 release 或旧 split。
先训 B，F 与最终展开器从头训练，0.8B 是目标规模；本步没有神经网络训练。
新政策是确定性参考程序，不是已学会的 B，也不是离线 jieba 的在线等价实现。

## 27 条夹具中最关键的几组

“夹具”是为检查某类错误而构造的固定输入。旧版有 **27 条合成输入**、**30 项单元测试**；
一个输入可被多项测试使用，测试也可临时构造额外小输入，两种计数不能混同。
以下解释针对旧 v0 合同，实际时刻与分组另存于最终验收报告的 `fixture_walkthrough`。

1. **`one-word`：末词与真实 EOF。** 输入 `antidisestablishmentarianism`，共 28 字符。
   时刻 28 已收到全部这些字符，但还不知道后面是否继续，`input_complete=false`、
   `document_ended=false`，可用流式关闭标签为 0。时刻 29 真正收到 EOF，两者变 true，
   才有 1 个关闭标签。若管线用文件长度在 28 就关闭，会偷看未来的“没有后文”。
   窗口结束或书籍摘录结束不能冒充这个 EOF。
2. **`same-prefix-a/b`：后文不能改变过去输入。** 二者先给出相同的
   `Store the identifier 3719.`，后文分别问末位 `final digit` 和首位 `first digit`。
   在共同前缀时，模型输入必须相同：只有已到达文字与收到的结束状态。问题、答案、
   全文标签和来源不能提前进入输入。两条同属一个隔离组且均在 train，避免相关变体
   跨 split。这检查数据可见性，**没有测模型能否记住 3719 或答出 9/3**。
3. **`chinese-role`：原文、坐标与监督权限。** `甲没有批准乙的申请。` 的离线参考是
   `甲｜没有｜批准｜乙｜的｜申请｜。`。其中“没有”是 cp `[1,3)`、UTF-8 byte `[3,9)`；
   两字六字节，不能混用坐标。时刻 3，输入只有“甲没有”，旧版中文流式标签为空，
   离线全文标签不进入模型输入。后句 `乙把17.3元交给甲。` 保留角色、方向与数值，
   但当前没有语义问答／否定理解／金额保真评分。
4. **`chinese-ambiguity`：离线答案不等于因果动作。** 旧版 jieba 对
   `研究生命起源` 给 `研究｜生命｜起源`，对 `南京市长江大桥` 给
   `南京市｜长江大桥`。这是固定版本的整段参考，不是人工金标。旧版不将汉字段强制成
   在线动作。本次另加的 FMM 政策实际产生 `研究生｜命｜起源`；这个反例直接说明
   有界和因果测试通过仍可能切错。二者分版本保存，不把差异悄悄抹掉。
5. **`inventory-10/17/73` 与 `near-a/b`：防止划分虚高。** 三个 inventory 输入只改
   盒中硬币数，模板族把它们放在同组、同一 train；不能拿“训练见过 10，测试换 17”
   当独立泛化。near 两条只把句末 `.` 换为 `!`，经 5-gram 近重复审计同组进入 test；
   变体保留。另有一条与英文原件完全相同的 `duplicate-english`，才被精确去重删除。
6. **`unicode` 与 `whitespace`：防止字符被悄悄改写。** `Cafe\u0301` 和 `Café` 看起来
   相近但原始 code point 不同；`𠀀` 是 1 code point、4 UTF-8 bytes、2 UTF-16 units。
   tab、CRLF、NBSP、前后空白和 emoji 原样保存，所有跨度应无空洞、无重叠并可拼回原文。
   这不承诺一个 emoji grapheme 就是一个单元。实际模型 token 对齐仍是另一个验收项。

这些夹具帮助发现数据工程错误；它们没有验证 B 的语义压缩能力或真实语料分词准确率。

## 独立真实候选与人工审核

4 条 MDN 连续原文摘录，共 **1072 code point**：英文 Unicode 371、UTF-8 403，
中文 Unicode 131、UTF-8 167。它们是短篇现代技术说明，不能代表广泛自然语言分布。
来源独立于旧 Alice／《論語》；英语与中文按规范化主题分成 **2 个平行内容族**。
中文源文件记录的英文 `sourceCommit` 原样保留，不假定它与当前英文版逐句一致。

- [固定来源及原文件](sources.lock.json)：固定仓库 SHA、URL、日期、原文件大小及 hash；
  [候选](sample_release/candidates.jsonl)另记精确提取 cp／byte 跨度、摘录 hash、原文及字节映射、贡献者历史与血缘。
- [许可通知](DATA-NOTICE.md)：MDN prose 为 CC-BY-SA 2.5，保留署名、来源、许可、改动说明及相同方式共享要求；
  两份完整上游许可随包，不混用夹具 CC0。权利未确认的源不得进入此清单。
- [候选审计](sample_release/candidate-audit.json)：对旧 **29 条输入／21 条 train** 做精确、
  归一化、数字模板、5-gram Jaccard ≥ .85 与显式来源／内容族审计，未命中；保存每条最近分数。
  这个字符审计不能排除未知语义改写。
- [隔离注册表](sample_release/holdout-registry.json)：所有候选 `training_eligible=false`、
  `split=null`，不进入 train／调参。注册表是未来接入时须执行的排除合同，不能自动约束尚未存在的训练管线。
- [审核说明](REVIEW-GUIDE.md)与 `sample_release/reviews/`：空 `units`、`pending`、`annotator=null`，
  人工填写无损分区、歧义及裁决。自动参考独立放在 `automatic-suggestions/`，不预填成金标。
  `candidates.validate_review` 拒绝未接受、未署名、未完整覆盖或有未决歧义的文件；它不能认证填写者身份。
  修改审核文件须另存新版本，不覆盖发布包。当前 **accepted gold = 0，无 gold accuracy**。

## 新的有界在线政策

[policy.json](policy.json)为 `prefix-fmm-v0.2`；固定 jieba 0.42.1 内置词典 hash，取纯 Han、
长度 1–6 的词，生成独立词表 hash。只利用词典条目，**不调用 jieba 的整段 DAG／HMM**。
到第 6 个可见 Han 字或真实非 Han 终止字符时，提交最长已知前缀，未知字回退为单字；
剩余汉字继续缓冲。6 是本次可审查的参考预算，尚未按人工质量或部署收益选优。

非 Han 块按旧英文规则在观察到分隔符时切分；普通块上限 32 code point，
以 `http://`、`https://`、`www.` 开头的块上限 64。达到上限立即分片，标
`budget_cap / continuation_possible=true`，不假装是完整词；后续片段重新识别类型。
URL 内遇中文转入 Han 政策，因此可能与旧版整段 URL 不同。空白与非 ASCII 标点／符号
逐个保留，不强求与旧版空白分组一致。已提交单元永不回改。

`Stream.feed(chunk)` 按字符到达计时，调用者须消费完生成器；网络 chunk 大小不改变动作。
`pause()` 不推进时钟、不 flush，用于窗口／摘录；`snapshot/restore` 继承实际 pending，
不得从标签初始化。`finish()` 只能在真正 EOF 到达时调用，时刻 N+1；结束后拒绝继续输入。
离线建议不被此 API 接收。输入 observations、参考 commits 和窗口／最终 states 分文件保存。
commit 含 cp／byte、decision／available／evidence／cutoff、原因和 committed 状态；state 含 pending 及其起点。

原文 pending 保守硬上界 **64 code point／256 UTF-8 bytes**；汉字暂存最多 6，处理后最多 5。
提交等待预算按首字符之后的到达步数计，保守 ≤64；压力样例实际最大 **63**，中文 **5**。
若输入停顿，墙钟等待可以无限长；这不是实时毫秒保证。固定词典、调用者输入 chunk、
观察／输出持久化存储不计入 pending；不得把这项界限扩张为完整模型常数内存。

实际处理旧 **26 条／2654 cp** 与新 **4 条／1072 cp**。旧组最大 pending 28 cp、
最长等待 28；新组为 12 cp／18 bytes、最长等待 12；真实英文摘录尾部保留共 22 cp 待续，
没有伪造 EOF。当前语料没有触发预算强切，专门压力例触发。
旧／新组与离线参考的边界对称差分别为 **46／8**，只比较已提交覆盖区间，
其中也含空白／URL 政策差异；称 **reference disagreement**，不是错误率或金标质量。

## tokenizer 实际盘点与运行边界

[盘点报告](validation/tokenizer-inventory-2026-11-03.json)检查 repo model-locks、models 下相关文件名、
默认 Hugging Face hub 模型目录名；未读凭证，也未扫描全盘。发现现有 Qwen3.6-35B-A3B GGUF
及 H-Net 配置／字节 tokenizer 源码，没有证据确定本轮所谓“3B 来源”。35B-A3B 的约 3B
激活参数不能当作 dense 3B 主干。最新执行仍为 B 先训，F／展开器从头训练。

Qwen 只解析前 **10,943,779 bytes** GGUF 元数据并保存其 hash：架构 `qwen35moe`、
词表 248320、merges 247587、`gpt2/qwen35` 元数据。没有读取 tensor 数据、加载大模型或运行
Qwen tokenizer；有词表元数据不代表 offsets／规范化已验收。先前锁中的完整权重 hash 保留，
本次不重读 36.9GB 权重。

实际运行现有 H-Net 官方 `ByteTokenizer`（代码提交 `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2`），
对 3 条 ASCII／中文／组合字符／非 BMP／emoji 短串验证 ids 等于 UTF-8 bytes，全部往返一致，
没有加载 H-Net 权重。其原生每字节 token 可能切开一个 code point，不能套用旧版严格
code-point 对齐适配器而宣称成功；这也不验证未来目标模型。目标配置保留
[model/tokenizer id、local path、revision、offset adapter 模板](tokenizer-config.example.json)，
尚未选择的项目明确留空，没有为了得到成绩另挑一个模型。

## 重跑与实际验收

依赖沿用旧 `.venv-word-boundary/`（Python 3.12.14，jieba 0.42.1），无需新包。
提交的六个源文件可完全离线重建；`acquire` 只是首次固定源获取工具，重新获取后 hash 不同会拒绝覆盖。

```powershell
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m training.word_boundary_v0_2.build --out models/word-boundary-v02-rerun
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m training.word_boundary_v0_2.verify --release models/word-boundary-v02-rerun --report models/word-boundary-v02-rerun-report.json
```

输出目录和报告都需尚不存在；日期为本次固定验收日期 2026-11-03。正式运行使用
`training/word_boundary_v0_2/sample_release` 和
[validation/run-2026-11-03.json](validation/run-2026-11-03.json)。
**原 30＋新增 16＝46 项测试全部通过**，包括不同 chunk 划分、同前缀不同未来、窗口状态恢复、
EOF N/N+1、Unicode 重建、长输入缓冲与延迟、禁止离线 DAG、故意篡改建议仍不影响
observations／commits／states、两次逐文件确定性构建与旧包／旧 split hash。
原 27 条输入的说明与观测证据存于报告 `fixture_walkthrough`。

额外 5 组压力例合计 **1,120,000 code point**，不保留输出历史。测得最大 pending
64 cp／244 UTF-8 bytes、最大提交等待 63 步。字典加载后开始 tracemalloc 的额外分配峰值
本次约 4.3KB；这是 Python 分配跟踪值，不是进程 RSS、词典或模型总内存。报告保存各组数据与耗时。

实际 H-Net tokenizer 盘点命令：

```powershell
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m training.word_boundary_v0_2.inventory --hnet-python 'C:\Users\Minzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' --out models/tokenizer-inventory-rerun.json
```

全程后台 CPU 小程序，没有租算力、部署／训练 LLM、LLM 标注、操作 KVM 或停止其它进程。
开始时分支干净，进程名快照未见常见 Python／llama／ollama／vllm 任务；不是对隐藏／远端任务的保证。
仍待人工审核、按质量选择词边界／缓冲政策、确定实际训练 tokenizer 及冲突策略、实现神经 B 消费器。
