# H-Net 完整 ROCm PyTorch 参考路径验收

日期：2026-11-03，America/Los_Angeles。专用分支 `codex/hnet-rocm-8060s`，起点 `0073a2d`。
实际在 Windows 11 / Ryzen AI Max+ 395 / Radeon 8060S 上完成官方 **874,315,264 参数**检查点的
完整 GPU 参考前向、全参数反向和一次优化器步。不是旧 70M encoder/router 探针，
也不是原生 Mamba／FlashAttention／Triton fused 内核已移植。

## 实际通过项与边界

- **passed：HIP 小算子。** `torch 2.9.1+rocm7.2.1`、`torch.version.hip=7.2.53211-158bd99533`；
  实际设备 `AMD Radeon(TM) 8060S Graphics / gfx1151`，可访问总内存 83,954,728,960 bytes。
  分配、填充、矩阵乘法、同步及 autograd 均执行，不以枚举代替运算。[preflight](results/preflight-2026-11-03.json)。
- **passed：完整模型 GPU 前向。** 两个编码／路由、最内层 **T26**、两级 dechunk／decoder、lm_head，
  328 个参数张量全部使用，无缺失 checkpoint key。FP32、batch=1，无 padding 或跨样本 packed 支持。
  50 字节原文 `The red box holds 17 coins. Keep it by the window.`，输入前 49 字节预测后 49 字节，
  实际长度 **49→21→8**；模块 trace 依次为 49、21、8、21、49，logits 为 `[49,256]`、有限且在 GPU。
  较早手工前向输入完整 50 字节时为 50→22→9；不要与正式 next-byte 验收混同。
- **passed：完整反向与 SGD 一步。** 没有冻结参数；328 个梯度全部存在、有限、非零且在 `cuda:0`。
  优化器 SGD，lr=1e-5、momentum=.9、foreach=false；328 个 momentum 状态覆盖全部 874,315,264 参数，
  均在 GPU 且有限。实测 head、内层 attention、两级 decoder、两级 router 权重都更新。
  FP32 EMA 方式 next-byte loss **1.34195876**，head 最大更新 **5.208e-6**，router 更新较小但非零。
  峰值 allocated **10,932,687,872 bytes（10.93 GB）**，含权重、梯度、优化器状态、激活与验收采样副本。
  原始检查点未写回，最终重算完整 SHA-256 与下载锁一致。[完整前后向](results/full-hip-training-2026-11-03.json)。
- **passed：源代码 BF16 dechunk 取整方式。** 参数仍为 FP32，dechunk 的 dt/x/B 及输出按原 forward
  的 BF16 转换参考计算；完整前向、全部梯度和一步更新也通过。loss **1.34148467**，峰值同上。
  这仍是 PyTorch 参考扫描，不是调用原 fused scan。[BF16 dechunk 运行](results/full-hip-bf16-dechunk-training-2026-11-03.json)。
- **passed：数值对照。** SSD、causal conv1d、SDPA、RMSNorm、dechunk 的 GPU FP32 前向／VJP
  对照独立 CPU FP64 递推／显式公式，容差 atol=rtol=2e-4；最大 VJP 差约 **4.06e-6**。
  STE 单独验证前向全 1、反向恒等。[算子报告](results/operators-2026-11-03.json)。
- **passed：完整模型 CPU 对照。** 同一 FP32 检查点，CPU 用 serial SSD、GPU 用 block SSD。
  预先固定 atol=rtol=1e-3；FP32 EMA logits 最大绝对差 **4.673e-5**，平均差 **4.862e-6**，
  loss 差 **2.384e-7**；两级边界位置一致。此样例路由到 .5 的最小距离约 .357／.149，
  没有近阈值点，不证明任意输入永不翻转。[FP32 对照](results/cpu-hip-parity-2026-11-03.json)。
  BF16 dechunk 方式最大绝对差 **.002062**，loss 差 **6.795e-6**，同一混合绝对／相对容差通过，
  两级边界也一致；不把 BF16 取整误差隐藏成逐位相同。[BF16 对照](results/cpu-hip-bf16-dechunk-parity-2026-11-03.json)。
- **passed：官方递归控制图对照。** 执行固定官方 HNet.forward、RoutingModule、ChunkLayer、STE
  的原始 AST，替换依赖 fused 内核的 Isotropic/dechunk 原语；CPU／GPU 的组合输出与本参考实现
  最大差约 **1.14e-5／1.91e-5**，边界一致。此项验证组合、残差和 padding，
  不是原 fused 算子对照。[CPU 图](results/official-graph-cpu-2026-11-03.json)、[HIP 图](results/official-graph-hip-2026-11-03.json)。

正式 FP32 GPU 前向约 7.31s、反向 1.46s、优化器步 1.00s；后运行的 BF16 dechunk 试验分别
约 2.21／1.47／1.11s。CPU serial 前向约 .41s、8 线程。含初始化／缓存差异且只有短输入单次试验，
不是速度基准；Python 参考的很多小 GPU 操作可能比 CPU 更慢，不能据此声称 ROCm 加速比。

## 原生内核、替代与缓存逐项状态

- **Mamba-2 SSD：参考路径 passed；原 fused Triton 路径 not-run。** 沿用已独立检查的
  `experiments/hnet/inspect_hierarchy.py` 的 block SSD；CPU serial recurrence 作数值对照。
  `mamba_ssm`、Triton 均未安装，未编译或运行原内核，不能写“原内核已移植”。
- **causal-conv1d：PyTorch HIP conv1d passed；causal_conv1d 扩展 not-run。** 保持因果 padding、
  depthwise 分组和原卷积权重；小算子前向与反向对照显式 unfold 公式。
- **attention：官方 PyTorch SDPA 的 MATH GPU 后端 passed；FlashAttention 扩展／AOTriton not-run。**
  正式脚本显式选择 MATH，保留因果／局部窗口、RoPE 和原 QKV 权重；没有打开实验性 AOTriton 开关。
  初次默认 SDPA 曾提示 gfx1151 efficient attention 是实验性的，后续参考不依赖它。
- **RMSNorm、gated RMSNorm、SwiGLU：可微 PyTorch GPU passed；原 fused 路径 not-run。**
  参数名、eps=1e-5、gate 顺序和残差保留；完整 backward 对所有相关参数检查。
- **路由、STE、chunk/gather：GPU passed。** 硬边界 `p>.5`，与原 argmax 的 .5 tie 规则一致；
  第一位置强制边界，selected probability 经原 STE 传梯度，不将离散索引假装成可微。
- **两级 dechunk/smoothing：两种参考方式 passed；原 fused scan not-run。**
  `ema-fp32-step` 使用官方 DeChunkLayer.step 的数学 EMA；`official-bf16-scan` 模拟 forward
  中 `dt=-log(1-p), x=h/dt, B=p, A=-1` 的 BF16 转换及 FP32 累积。
  模型浮点语义和这种显式取整分开报告，均未与不可用的原 fused 内核作直接对照。
- **增量推理缓存：not-run／未实现。** 本参考只做完整前缀 forward；没有实现或验收官方 Mamba
  conv/SSM 状态、attention KV、router/dechunk 缓存和 GenerationMixin。不能把 prefill 成功写成
  高效缓存生成已通。没有声称动态批处理、长训练、mixed-precision 主干或性能优化完成。

所有模型数值运算在声明的设备；输入、参数、输出、梯度、优化器状态均检查设备。
仅分支长度及诊断结果取到 CPU，CPU gold 和误差比较另行运行；无数值 CPU 静默回退。

## 当前支持文档、版本选择和真实失败

不能把旧 `radeon-ryzen/en/latest` 当成最新完整支持信息：该页只覆盖到 **7.2.1**。
已核对[统一 ROCm 10.1 矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
及其 HTML selector，包含 8060S/gfx1151、Windows 11 25H2；
[新 PyTorch 安装页](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
有 `torch[device-gfx1151]==2.14.0+rocm10.1.0` 入口。实际官方 Windows cp312 wheel HEAD 返回 200，
大小 115,562,766 bytes，没有下载／安装它。

新矩阵对应 Adrenalin 26.10.41.05，
[驱动说明](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html)
给出 Driver Store **32.0.31041.5005**；本机 **32.0.31032.1003** 较旧。
因此本轮保留已经完整实测通过的独立 7.2.1 环境，将 10.1 标 not-run；
没有由此断言新版不能运行或更改驱动。若之后采用新版，先核验这台设备适用的 OEM／AMD 驱动包并获得系统变更确认。
[兼容性记录](results/compatibility-review-2026-11-03.json)。

旧版 7.2.1 [支持表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)
列出 Max+395/gfx1151 和 Python 3.12，
[安装页](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html)
提供本次官方 wheel。其[限制页](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsryz.html)
的“No ML training support”只按该版本的官方支持边界理解。本次 full backward/SGD 是实际通过的
实验参考路径，不扩张为 AMD 对这个栈的生产训练支持；更不能把旧限制推给所有新 Windows 版本。
Mamba 官方当前 prerequisites 写 Linux，FlashAttention 则有 AMD CK／Triton 路径；
这些资料不能代替本机完整原生 fused 链验证，也不能据此断言永远不可能移植。

新环境首次导入因 **MSVCP140.dll 不在搜索路径**失败，并非 AMD DLL 文件缺失。
PE 依赖逐项加载定位后，通过进程内预加载已有 Codex Poppler 自带、Microsoft 签名有效的
MSVCP140 14.51.36247.0 解决；不复制或分发 DLL，不改系统。首次在受限执行中 GPU 分配出现
HIP DLL `0xC0000005`，允许设备访问后同形状 randn256／matmul／backward 通过。
差异符合沙箱设备访问问题，未独立定位驱动内部原因；没有关闭安全功能。
[初始失败与处理](results/startup-failures-2026-11-03.json)、[硬件／签名](results/environment-2026-11-03.json)。

全程未修改驱动、WSL、BIOS、安全设置，未重启、操作 KVM、停止其它进程或使用付费算力。
开始和结束命名进程快照未见其它常见训练进程；快照不能保证所有隐藏或远端任务。

## 环境、来源与重跑

隔离环境 `.venv-hnet-rocm/`；旧 `.venv-hnet/` 和数据环境保持原样。
[依赖版本锁](requirements-rocm.lock)与[安装包／关键 DLL 指纹](results/environment-lock-2026-11-03.json)
保存精确版本、官方来源、安装 RECORD／二进制 hash；不是 hash 强制的原 wheel 归档。
SDK 内 amdclang++／hipcc 的 `--version` 实际通过，版本为 clang 22.0.0git／HIP 7.2.53211；
没有编译 H-Net 扩展。[编译器记录](results/compiler-versions-2026-11-03.json)。

检查点沿用 [hnet-2stage-l.json](../../model-locks/hnet-2stage-l.json)：官方权重 revision
`eb0db7323b1131e6bdb8d1c7367d3f815d09426c`，874M、3,497,374,019 bytes，
SHA-256 `5711de80965d7ec8430a2db05a4859bc0700576e3e7841658fe1640d0c0e011c`。
[源码锁](source-lock.json)重新下载核对 hnet.py、dc.py、mamba2.py 与本地副本逐字节一致，
固定 H-Net `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2`、Mamba `a6a1dae6efbf804c9944a0c2282b437deb4886d8`。
H-Net MIT 通知在 [UPSTREAM-LICENSE.txt](UPSTREAM-LICENSE.txt)；权重／环境不提交 Git。

首次安装命令，已有环境不重复安装：

```powershell
& 'C:\Users\Minzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv-hnet-rocm
& '.\.venv-hnet-rocm\Scripts\python.exe' -m pip install --no-cache-dir -r experiments/hnet_rocm/requirements-rocm.lock
```

运行者必须允许该进程访问 HIP 设备。本机无需修改系统；`bootstrap.py` 默认从现有 Codex runtime
预加载 MSVCP140，也可用进程变量 `HNET_MSVC_DLL` 指向已有可信 x64 运行库；
正常安装 Microsoft 官方 VC runtime 的机器可以直接采用其搜索路径。不要将系统安全设置的关闭作为本验收前提。
报告拒绝覆盖，重跑时使用新的 `models/` 输出名：

```powershell
& '.\.venv-hnet-rocm\Scripts\python.exe' -X utf8 -m experiments.hnet_rocm.validate --phase preflight --out models/hnet-rocm-preflight-rerun.json
& '.\.venv-hnet-rocm\Scripts\python.exe' -X utf8 -m experiments.hnet_rocm.validate --phase ops --out models/hnet-rocm-ops-rerun.json
& '.\.venv-hnet\Scripts\python.exe' -X utf8 -m experiments.hnet_rocm.validate --phase full --device cpu --scan serial --out models/hnet-rocm-cpu-rerun.json
& '.\.venv-hnet-rocm\Scripts\python.exe' -X utf8 -m experiments.hnet_rocm.validate --phase full --device cuda --train --out models/hnet-rocm-gpu-rerun.json
& '.\.venv-hnet-rocm\Scripts\python.exe' -X utf8 -m experiments.hnet_rocm.validate --phase compare --cpu-report models/hnet-rocm-cpu-rerun.json --gpu-report models/hnet-rocm-gpu-rerun.json --out models/hnet-rocm-parity-rerun.json
& '.\.venv-hnet-rocm\Scripts\python.exe' -X utf8 -m experiments.hnet_rocm.official_graph --device cuda --out models/hnet-rocm-official-graph-rerun.json
```

BF16 dechunk 对照对两边 full 命令都加 `--dechunk-mode official-bf16-scan`，再 compare。
环境指纹：`python -m experiments.hnet_rocm.lock_environment --out models/hnet-rocm-env-rerun.json`。
正式命令使用本目录 `results/` 对应文件名；每份报告记录源码指纹、输入、设备、损失、时序、误差和范围。
没有保存更新后的模型，实验结束后重跑仍从原权重出发。
