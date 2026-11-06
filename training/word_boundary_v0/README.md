# 第 0 步：中英文首层固定词边界

这是已运行的小规模数据准备原型，不包含模型训练。当前执行口径为：**先训练 B 规约器；F 和最终展开器从头训练；0.8B 是目标参数规模，不是复用 Qwen 0.8B 主干。** 此口径由用户后续明确，优先于历史记录中的预训练 F 继承方案；[v1.2 原文](../../path_control_pretraining_part1_v1.2.md)仍保真保存，本步沿用其第 10 节的来源、可见性和监督隔离要求。

## 已交付与适用范围

- [固定配置](rules.json)、[依赖锁](requirements.lock)、[来源与词典锁](sources.lock.json)。
- [输入原文](inputs/documents.jsonl)与[provenance](inputs/provenance.json)，真实摘录和合成夹具明确分开。
- [已运行的数据包](sample_release/manifest.json)：原文、字符／字节区间、全部参考边界、延迟流式标签、独立 Observation/Supervision 文件、切片、split／去重审计、token 冲突及统计。
- [逐项可读样例](sample_release/examples.txt)、[实际测试报告](validation/run-2026-11-03.json)、[设备及环境检查范围](validation/environment-2026-11-03.json)。

29 条输入包含 2 个真实原作的摘录和 27 条自写夹具；剔除 1 条精确重复，隔离 2 条空／全空白记录，保留 26 条，切成 37 片。train/dev/test 为 21/2/3 条文档、18/2/2 个隔离组。两条真实摘录均在 train，dev/test 只有夹具；不据此报告真实语料质量或泛化成绩。没有为了让真实文本落入指定 split 而调种子。

输出是首层参考边界和消费权限的试制件，**不是完整 v1.2 D1/D2**：未定义／执行神经 B、F、父态 latent 或 B⁻ RETURN，未实现真正模型 tokenizer，也未获得中文流式政策或人工金标质量验收。

## 固定来源与许可

真实文本使用 *Alice's Adventures in Wonderland*（Lewis Carroll，ebook 11，所取版本更新 2025-06-26）和《論語》（传统归于孔子，ebook 23839，版本更新 2014-01-27），各取前三个选定段落，分别保留 1360 和 138 个 Unicode code point。获取日为 2026-11-03 America/Los_Angeles，完整源文件总计约 263KB，存于忽略的 `models/word-boundary-sources/`；只提交小摘录、提取区间、来源 hash 和许可。

两个源 URL 会随出版方更新，**版本由 SHA-256 锁定，不把 URL 当永久不变的内容**。重新获取内容若不符会失败，不能静默使用新版本；提交的摘录可离线重跑。`prepare_sources.py` 只选择连续原文跨度，不规范化、改写、翻译或统一换行。英文保留原弯引号与 CRLF，中文保留传统字形；后者不能代替现代简体中文的真实质量样本。

原作目录标明美国公版；此结论不扩张为全球权利保证。来源署名、许可链接及完整文本随[数据通知](DATA-NOTICE.md)和 `licenses/` 提供，并复制到数据包；再分发时保留这些文件和 provenance。夹具为本次任务自写／合成，标 CC0-1.0，不冒充真实语料或人工分词金标。jieba 软件与随包词典按 MIT 许可保存；原文许可不因使用 jieba 而改变。

许可快照保留来源 CRLF、空行及尾空格，不进行格式清洗；仅这两处第三方许可目录从 Git whitespace 检查中豁免。代码、JSON 和项目说明仍正常检查，数据 hash 不受行尾自动转换影响。

## 英文、中文及非词字符规则

`boundaries.segment` 生成覆盖全文的互不重叠 `[start,end)` 单元。Unicode code point 是逻辑坐标，UTF-8 byte offset 是对应物理坐标；`cp_to_byte` 包括末端偏移。这里不以 JavaScript UTF-16 长度冒充字符数，不做 NFC/NFKC、大小写或空白转换。

- 连续 Unicode 空白单独成单元，原样保留空格、tab、CR、LF、NBSP；流式标签不为这些分隔单元额外训练边界，离线参考保留其跨度。
- 英文／非汉字母构成字串，组合附加符保留在串内。`can't`、`can’t`、`re-enter`、`re‑enter` 保持一单元，仅内部且后接字母的撇号／连字符连接。前后引号、尾随 `dogs'` 的撇号分开；em dash 不作内部连字符。
- 缩写清单固定在 `rules.json`，如 `Dr.`、`Mr.`、`e.g.`；`U.S.`、`p.m.` 这样的多段 initialism 保持整体。其他句点一般为独立标点；句末统计只是 `.!?。！？` 独立标点候选，不声称语义分句或 EOS 判定。
- ASCII 数字允许正负号、逗号三位分组、小数和指数；`-17.30`、`1,234.50`、`6.02e23` 为一单元，`v1.2` 切为字母与数字，`73/37` 中 `/` 分开。不是通用数值解析器，完整语义另验收。
- URL 限 `http://`、`https://`、`www.`；终止于空白、角括号、双引号或中文 `，。；：！？`，再剥离固定尾标点集合。中文路径可保留；ASCII `)` 等尾字符一律按本版本规则剥离，不做 URI 括号配对。email 无专用规则。已加入中文逗号紧随 URL 的回归例。
- 连续 Han 字符段用独立 jieba Tokenizer、精确模式、`cut_all=False`、`HMM=False`、无用户词典／动态加词／并行／paddle。包固定 0.42.1，内置 `dict.txt` SHA-256 为 `7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8`，不符即拒绝。
- 其他标点和符号保留为独立单元；emoji ZWJ 序列不承诺一个 grapheme 一个单元，仍保证原文重建。

这是版本化的正字法规约，不是“词”的唯一语言学定义。`unit_boundary_rate` 包含空白／标点／符号；`lexical_end_rate` 只统计字词、数字和 URL，不能混用。长度直方图分别给 code point／byte、全单元／词类，中文、英文、混合文本分别报告，并另列真实语料与夹具数量。

## 离线参考与流式动作

`events.compile_tapes` 输出两个分离文件。Observation 只有逐字符到达和外部实际收到的结束信号，`model_input` 仅白名单读出 `text / input_complete / document_ended`。来源、split、全文长度、参考边界、未来标签、latent 和答案都不进入该输入。训练消费器按**观察事件时钟**调用，移除标签不会改变模型输入序列。

`READ_FULL` 参考标签在完整输入接收后才可消费；对 N 字符，结束信号在 N+1。该模式有合法完整文本条件，不能拿它去预测已读内容的早期位置并声称流式 LM。中文 jieba 的路径选择可能依赖整段后文，汉字段在本版**没有可强制的 READ_STREAM 动作**。zh 文档中独立的 ASCII 段（如段落编号）仍可走下面的非 Han 政策，不能据此称中文流式分词已实现。

英文流式基线是保守的**缓冲块延迟参考**：等一个非 Han 的非空白块后面第一个空白真正到达，才对缓冲块的内部边界／非边界计损失并允许当时执行。块内位置 `p` 的边界不回写为发生在过去 p 时刻的动作；记录 `decision_at / available_at / evidence_available_at / observation_cutoff / apply_loss_at`。末词必须等真实 EOF 信号 N+1 才可 flush；在 N 时刻与同前缀继续增长的文档保持一致。

本原型未给非空白块设置流式最大长度，因此这是**延迟分块基线，尚不是满足有限开放 buffer 的部署政策**。连续中文块直接 abstain，不偷用离线边界；没有实现自由 B 路由或参考动作的神经执行。LLM 标注仅保留为有人工金标后再研究的选项，本步未调用标注服务。

真实摘录以 `source_excerpt_end` 标明可用输入结束，不假装书籍 EOF；固定 128-code-point 切片也不注入 EOS。切片由原文位置决定，不由词标签决定；跨窗单元被记录并 mask，完整文档参考另走 READ_FULL。消费窗口必须合法重放前缀或继承真实前置状态，不能从标签构造初始状态。

## 文档划分与去重

先在原始文档图上合并来源文档、显式内容族、模板族、平行／派生关系，再计算 split，最后切片。同一本书的摘录／后续片段共享来源文档组；显式派生引用不解出则报错，ID 冲突也报错。

精确原文重复只留一个代表，优先保留无派生依赖的原件。近重复审计仅为匹配计算 NFKC＋casefold＋空白归并，原文保持原样；长度至少 30 时，字符 5-gram Jaccard ≥ .85 的对连为一组；相同的数字模板签名也分组。近重复及模板变体保留在同一 split，不自动当同义句删掉。所有连边、分数、代表、丢弃／隔离原因和归属见 `split-audit.json`；它们可能重叠，不能把连边计数相加当作重复文档数。

使用固定种子对隔离组 ID 的 SHA-256 排序，按组数近似 80/10/10，至少三组时 dev/test 各至少一组；这是按组分配，文档数不承诺恰好 80/10/10。增加源文档会改变排序，需要新版本与新 manifest，不默默覆盖旧 split。二次复杂度只支持 ≤5000 文档；未知跨语言翻译／语义改写不是该字符审计能保证检测的，扩量前仍需更完整血缘和语义审计。

## 模型无关 token-offset 接口

`alignment.align_tokens(text, boundaries, tokens, offset_unit=..., tokenizer_id=..., original_text=...)` 接收 `{id, span:[start,end], special, surface?}`。坐标必须明确为 `unicode_code_point / utf8_byte / utf16_code_unit`，偏移针对原文。`surface` 是源文本片段，不是 `Ġword` 一类词表拼写；若只有 offset，报告 `adapter_geometry_only`，不能声称已验证解码一致。

显式报告并保留原边界：一个 token 内有词边界、多个边界共用 token、覆盖空洞／重叠／乱序、零长或反向 span、UTF-8 字符中间偏移、UTF-16 半 surrogate、special token 占真实文本、surface 或规范化文本不符。**不吸附到邻近 token。** 有冲突的标签 mask，调用方可整条隔离；本步未选择任何生产 tokenizer，数据包 `training_readiness.real_model_tokenizer=false`。

输出仅用“一 code point 一 token”和“整段一 token”两个人工适配器检查接口：前者可对齐，后者专门产生冲突。它们不是 Qwen、3B 或其它模型 tokenizer 的实测。

## 安装与重跑

已测环境：Windows、Python 3.12.14、pip 25.0.1、jieba 0.42.1、Unicode 数据库 15.0.0。只安装 jieba，sdist 哈希锁定；安装器构建的 wheel 字节不声称跨机器完全一致，运行数据由固定规则、词典与 manifest 记录的代码／环境验收。

在仓库根目录，已有输入可离线构建。环境首次安装需网络：

```powershell
& 'C:\Users\Minzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv-word-boundary
& '.\.venv-word-boundary\Scripts\python.exe' -m pip install --require-hashes --no-deps -r training/word_boundary_v0/requirements.lock
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m training.word_boundary_v0.build --out models/word-boundary-rerun
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m training.word_boundary_v0.verify --release models/word-boundary-rerun --report models/word-boundary-rerun-report.json --client-date 2026-11-03
```

`--out`／报告需是尚不存在的新路径，日期使用实际用户本地日期。构建器拒绝覆盖已有目录。若要重新获取并复核原始完整源文件，可单独运行：

```powershell
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m training.word_boundary_v0.prepare_sources --cache models/word-boundary-sources --fetch
```

检查少量模型可见输入与独立监督：

```powershell
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m training.word_boundary_v0.inspect_sample --release training/word_boundary_v0/sample_release --doc-id fixture:chinese-role --cutoff 3 --mode READ_STREAM
```

此时模型输入只有“甲没有”及两个 false 状态，没有参考边界或 loss target。完整样例输出另见 `examples.txt`。

## 实际验收与下一步

最终实际执行 `build --out training/word_boundary_v0/sample_release`，再执行 `verify --release training/word_boundary_v0/sample_release --report training/word_boundary_v0/validation/run-2026-11-03.json --client-date 2026-11-03`。**30 项测试通过**，含两次独立构建的所有文件逐字节比较、输入／输出 manifest 校验、100 个固定随机字符串的无损分割、近重复／split／切片隔离、Unicode／对齐冲突、未来／标签隔离、标签不改变观察时钟以及真实 EOF 对照。测试不运行神经网络，不代表真实语料分词准确率。

原型确实运行在 CPU：现场只读核查 Ryzen AI Max+ 395、8060S、约 128GB 内存；核查时未见 Python/llama/ollama/vllm/train 命名进程，不代表审计了所有远端／隐藏计算。未停止其它进程、操作 Mac KVM、验证 ROCm/NPU、部署 LLM 或租用算力。

下一步阻塞在中文可执行流式政策及最大 buffer／延迟预算、现代自然文本的独立金标评估、实际 tokenizer 冲突处理配置、以及神经 B 消费器。规则基线和这批夹具先作为回归契约保留，再扩大真实语料；不把这些缺口掩盖成模型训练已准备完毕。
