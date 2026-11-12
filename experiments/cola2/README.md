# Cola2 官方权重本机试跑

试跑于 2026-11-11 晚开始，正式 checkpoint 于 **2026-11-12 00:06（太平洋时间）** 跑通。对象是 [arXiv:2610.31305](https://arxiv.org/abs/2610.31305) 对应的 ByteDance-Seed Cola2 base 模型；没有使用其他同名模型。

权重来自[官方 Hugging Face 发布](https://huggingface.co/ByteDance-Seed/Cola2/tree/db876b905cf719b60da33c12a75021ac111f7a67)，revision `db876b905cf719b60da33c12a75021ac111f7a67`；源码来自[官方 GitHub 固定提交](https://github.com/ByteDance-Seed/Cola2/tree/085817168c9d6a718812fbc0075b968fc4cd5f9f)。模型仓库公开、无需登录，八个 FP32 safetensors 共 **35,072,206,700 bytes**，与作者 SHA256SUMS 及 Hub LFS SHA-256 全部一致。只取推理文件，没下载训练或评测集。[资产锁](assets.lock.json)与[验证回执](results/download-verification.json)分别记录预期与实际。代码是 Apache-2.0；模型卡未填独立权重许可证，且官方[模型卡许可段](https://github.com/ByteDance-Seed/Cola2/blob/085817168c9d6a718812fbc0075b968fc4cd5f9f/docs/model_card.md#license)明确区分权重、tokenizer、数据与代码。本轮只作本机研究，不再分发权重。

## 真实生成

采用[官方 quickstart](https://github.com/ByteDance-Seed/Cola2/blob/085817168c9d6a718812fbc0075b968fc4cd5f9f/examples/quickstart.py)的原提示、BF16 及 `generate()` 设定；以固定官方源码类离线加载，避免让 Transformers 再动态加载远端代码。Hub 和 GitHub 的两份模型计算文件 Python AST 相同；配置文件差异是 GitHub 有适配不同 Transformers `validate()` 签名的逻辑。保留发布预设：每轮 8 步 low-noise Euler、DiT CFG 10、decoder CFG −0.9、每个解码 chunk 一个 span、时长预测和回编码开启、greedy token、最多 32 个新 token。只额外固定 latent RNG seed `42`，且明确选择 PyTorch SDPA MATH 后端。

输入：`Question: What is the capital of France? Answer:`

真实续写（保留开头空格、重复和错误）：

```text
 Paris. Question: Who is the the the president of France? Answer: Nicolas Harko Sark: Question: Who is France's Prime minister? Answer:
```

这是一条 base 模型续写，开头答对，后面不能当作事实。11 个提示 token，32 个返回 token；内部统计为 **2 轮生成、16 次去噪步、2 次时长预测、16 个解码 chunk、2 次回编码反馈**。生成 **7.54 秒**，权重加载、移到 GPU 和准备输入 **14.90 秒**，均为一次同步墙钟观测，不是吞吐基准。[原始运行记录](results/checkpoint-hip-bf16-v2.json)含 token IDs、实际设置和内存。

## 文字 → latent → 文字

原文：`The red box holds 17 coins. Keep it by the window.`

官方 `encode()` 将 15 个 token 编成 **9×64** 的 BF16 latent。span 时长 `[1,3,3,1,1,1,3,1,1]`；文字分组 `The | red box holds | 17 | coins | . | Keep | it by the | window | .`（展示时省略了部分前置空格，精确跨度见原始记录）。节点比 `9/15=0.60`，不是存储位率，也不是句子级瓶颈。

- 官方 `forward()` 的重建 logits 逐位 argmax：**15/15 token 相同**。这个路径每个 span 可看此前原文，不能单独证明 latent 保真。
- 用官方 VAE 的 `expand_and_refine` / `span_query_logits`，每个 span 只看此前**模型输出**，使用编码给出的时长，逐位恢复同一句话。时长仍是额外信息。
- 从空前缀开始，同样只接自己的输出，改用 `latent_decode` 从 latent **自行预测时长**：预测时长也完全相同，输出仍是 `The red box holds 17 coins. Keep it by the window.`，**15/15 token 相同**。这个探针使用纯条件 logits（decoder CFG 0）；是官方组件的短重建实验，不是把正式生成的 CFG −0.9 改掉，也不是 upstream 独立公开的 `decode()` 命令。

三条重建路径与 span/latent 取样合计 **2.07 秒**。仅一条原文，不证明一般可逆、长上下文保真、或跨域语义压缩能力。生成样例本身的重复、名字错误也显示二者不能互推。

## 规模和环境

- **8,768,019,912 参数**，运行时全在 `cuda:0`（PyTorch 的 ROCm 设备命名），全为 BF16。1,137 个 checkpoint 张量含 `vae.segmenter.rate_bias` 一个 FP32 持久缓冲，因此磁盘逻辑元素比参数多 1。检查了实际张量形状、分片索引和范围；加载无缺失、额外或形状不符的 key。
- **4,038,996,032** 参数属于 34 层 latent DiT；**4,022,620,032** 属于 36 层 Qwen3-4B 形状的 token decoder（含词表输出层）；其余 **706,403,848** 属于编码、桥接、时长和条件模块。这里的 34/36 是网络深度，非规约层数。
- Ryzen AI Max+ 395／Radeon 8060S `gfx1151`／标称 128 GB 统一内存，Windows 11 build 26200，驱动 `32.0.31032.1003`，无 WSL。复用 Python 3.12.14、Torch `2.9.1+rocm7.2.1`，叠加 Transformers `4.57.6`、safetensors `0.8.0` 和 accelerate `1.15.0`，不修改旧环境。模型、临时文件、pip 缓存和新依赖均在 D 盘。
- 生成时 PyTorch 峰值 allocated **17.84 GB**、峰值 reserved **18.02 GB**；进程峰值 working set **26.37 GB**。统一内存口径有重叠，不能相加；这些不是全机／驱动峰值。CPU 只参与加载和调度，未用 CPU 替代模型运算。

## 复跑

本机已安装且已校验，进入项目目录后可直接复跑（`run-id` 每次不同）：

```powershell
Set-Location 'D:\ai-modeling'
& '.\.venv-hnet-rocm\Scripts\python.exe' -X utf8 -m experiments.cola2.run --device cuda --dtype bfloat16 --run-id replay-01
```

如需在同结构环境重新准备本地资产，从项目根目录执行：

```powershell
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m experiments.cola2.acquire source
& '.\.venv-word-boundary\Scripts\python.exe' -X utf8 -m experiments.cola2.acquire model --workers 3
$cola = Join-Path (Get-Location).Path 'models\cola2'
New-Item -ItemType Directory -Force -Path "$cola\tmp", "$cola\pip-cache" | Out-Null
$env:TEMP = "$cola\tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$cola\pip-cache"
& '.\.venv-hnet-rocm\Scripts\python.exe' -X utf8 -m pip install --index-url https://pypi.org/simple --no-deps --target "$cola\deps" -r experiments/cola2/requirements-overlay.txt
```

### 已核查的启动边界

仅加 `PATH` 的首次 ROCm 探测遇到 `amd_comgr0702.dll` 依赖加载失败及 offload-arch 应用控制警告；复用项目原有的可信 MSVC 进程内预加载后，BF16 SDPA 及完整模型均实际通过。未改应用控制、驱动或系统运行库。`tiny-hip-bf16.json` 是随机权重全链 preflight，不是正式模型。首次正式尝试在本地元数据检查中错误地把持久缓冲计入 HF 参数数，未开始加载权重；失败记录保留，第二次按缓冲和参数分开验证后通过。Transformers 的通用 regex 警告保留；这两条输入的快慢 Qwen tokenizer ID 完全相同，没有改 tokenizer，也没有推广到其他输入。GenerationMixin 警告未阻止 Cola2 自己的 `generate()`；没有尝试 Transformers 5 或 CPU 正式权重推理。
