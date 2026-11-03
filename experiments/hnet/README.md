# H-Net 层级节点率探针

使用官方 `cartesia-ai/hnet_2stage_L` 的 FP32 权重，运行两级 encoder/router。固定版本、大小和 SHA-256 见 [模型清单](../../model-locks/hnet-2stage-l.json)。权重位于 `models/hnet_2stage_L/`，不进入 Git。

## 执行范围

- 原权重共 874,315,264 参数；探针实际使用 70,114,048 个 encoder/router 参数。
- 根据固定版本官方代码，用普通 PyTorch 实现 Mamba-2 SSD、RMSNorm、SwiGLU、RoPE 和 SDPA，替代 CUDA/Triton fused kernels。
- 只执行决定边界所需的读入路径；不运行最内层 LM、dechunking 或 decoder，不生成文字，不测生成质量。
- 原始 UTF-8 字节，无 BOS/EOS，每条样本重置状态。统计节点数和压缩率，不把连续 latent 当离散码本，也不把节点率当成 bits/byte。
- 已检查 SSD 的分块／逐步计算一致、实际权重下边界一致、追加未来后缀不改变前缀边界、单字节边界，以及官方路由源代码与探针对齐。未验证与官方 CUDA fused kernels 的数值一致性。

## 本机环境

可用下列脚本下载固定版本权重、配置和检查用源码快照；已有权重会先校验，下载中断可续传：

```powershell
& '.\experiments\hnet\download.ps1'
```

Windows 11 Pro build 26200；Ryzen AI Max+ 395 / Radeon 8060S；显示驱动 32.0.31032.1003。运行环境为 Python 3.12.14、PyTorch 2.8.0+cpu、FP32、8 个 CPU 线程。`.venv-hnet/` 排除 Git。

当前机器的系统 `python` 是 Windows 应用别名，创建环境时实际使用：

```powershell
& 'C:\Users\Minzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv --system-site-packages .venv-hnet
& '.\.venv-hnet\Scripts\python.exe' -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
```

PyTorch 在默认工具沙箱内曾无法载入 DLL，允许执行后可正常运行；本轮未安装系统 VC++、显卡驱动或 WSL，未部署 ROCm。CPU 参考探针的选择不代表 Max+ 395 不支持 ROCm。

## 重跑

从项目根目录运行：

```powershell
& '.\.venv-hnet\Scripts\python.exe' experiments/hnet/inspect_hierarchy.py
```

结果保存在 [hierarchy.json](results/hierarchy.json)，包含完整输入、各级选择概率、字节索引、跨度和检查结果。`--model-dir`、`--output`、`--threads` 可覆盖；`--device cuda` 可供兼容的 CUDA/ROCm PyTorch 环境使用，但本轮没有测试 GPU。

交互报告从保存的结果生成，不会重新推理：

```powershell
& '.\.venv-hnet\Scripts\python.exe' experiments/hnet/build_report.py --output '<可写的绝对路径>\hnet-level-rates.html'
```

该文件是会话内视图片段，显示等长的五个冗余条件，并支持查看两级字节切分；同时生成 [rates.csv](results/rates.csv)。源码模板和数据都在仓库内，不依赖网络。

## 解释边界

各级选择位置投影到原文，只说明调用／更新发生在哪些字节位置。原 H-Net 使用 lookahead 式边界，每个 encoder 状态也含因果历史；显示跨度不是独立可解码的语义码。

冗余组都为 560 字节：完全重复句子、只改数字、更换等长字段、不同内容散文前缀、随机 ASCII。前三项共享模板，较适合局部对照；散文和随机串同时改变分布与预测结构，不能仅按它们给出严格信息熵排序。样本是人工小样本，不是模型总体基准。

少量粗层节点不证明丢失所有输入，也不证明高压缩质量好。应另测后续精确引用、任务质量和路径干预。尤其独立随机的过去未必帮助预测下一字符，这与稍后必须复述随机编号的任务不同。
